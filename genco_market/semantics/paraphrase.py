"""Cached LLM paraphrase of template text.

- Applied at L2/L3 only. The LLM disk cache (agent/llm.py, keyed by model,
  effort, schema and messages) makes replays deterministic; ``offline=True``
  never calls the API and keeps the template text on a cache miss.
- The prompt differs from the agent's perception prompt (no "analyst" role,
  no extraction), so the agent does not grade its own homework.
- A paraphrase is rejected (template text kept) unless every number, time,
  zone letter and "f-t" branch name survives exactly (``keeps_facts``).
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

# order matters: f-t names and times before bare numbers
_KEEP = re.compile(r"\b\d+-\d+\b|\b\d{1,2}:\d{2}\b|\b\d[\d,]*(?:\.\d+)?\b|\bZone [A-E]\b|\bt=")

PROMPT_VERSION = 2
SYSTEM = """You rewrite short power-market bulletin items so they read like natural text from the given source
(ISO notice, weather service, news wire, social media post, or trader chat).
Hard rules:
- Keep every number, time (HH:MM), date, zone name ("Zone C"), line name like "38-37", bus number and "t=..." interval tag exactly as written.
- Do not add, drop or change any fact, number, probability word or hedge strength.
- Keep the source's register (social posts stay informal, ISO notices stay formal).
- Vary vocabulary and sentence structure noticeably; do not just copy the input.
- One or two sentences. Output only the rewritten text."""


class Paraphraser(Protocol):
    def __call__(self, text: str, *, item_id: str, seed: int, source: str = "") -> str: ...


class IdentityParaphraser:
    def __call__(self, text: str, *, item_id: str, seed: int, source: str = "") -> str:
        return text


def fact_tokens(text: str) -> list[str]:
    return sorted(_KEEP.findall(text))


def keeps_facts(original: str, paraphrased: str) -> bool:
    """True if numbers, times, zone letters, branch names and t= tags are preserved."""
    return fact_tokens(original) == fact_tokens(paraphrased)


class ParaphraseOut(BaseModel):
    text: str


@dataclass
class ParaphraseStats:
    n: int = 0
    rejected: int = 0  # fact check failed
    errors: int = 0  # API / budget errors
    misses: int = 0  # offline cache misses
    examples: list[dict] = field(default_factory=list)


class CachedLLMParaphraser:
    def __init__(self, llm=None, *, offline: bool = False, keep_examples: int = 500):
        self._llm, self.offline, self.keep_examples = llm, offline, keep_examples
        self.stats = ParaphraseStats()
        self._lock = threading.Lock()

    @property
    def llm(self):
        if self._llm is None:
            from genco_market.agent.llm import LLM

            self._llm = LLM(model=os.environ.get("OPENAI_MODEL_TEXTGEN"), effort="low", timeout_s=30)
        return self._llm

    @staticmethod
    def user_prompt(text: str, seed: int, source: str) -> str:
        return f"prompt_version: {PROMPT_VERSION}\nvariant: {seed}\nsource: {source}\ntext: {text}"

    def __call__(self, text: str, *, item_id: str, seed: int, source: str = "") -> str:
        user = self.user_prompt(text, seed, source)
        out, status = text, "ok"
        try:
            if self.offline and not self._cached(user):
                status = "miss"
            else:
                cand = self.llm.parse(SYSTEM, user, ParaphraseOut, purpose="paraphrase", cache_mode="readwrite").text.strip()
                if cand and keeps_facts(text, cand):
                    out = cand
                else:
                    status = "rejected"
        except Exception:
            status = "error"
        with self._lock:
            st = self.stats
            st.n += 1
            st.rejected += status == "rejected"
            st.errors += status == "error"
            st.misses += status == "miss"
            if len(st.examples) < self.keep_examples:
                st.examples.append({"item_id": item_id, "source": source, "status": status, "before": text, "after": out})
        return out

    def _cached(self, user: str) -> bool:
        llm = self.llm
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        return llm._cache_get(llm._key(msgs, ParaphraseOut, llm.effort), "read") is not None
