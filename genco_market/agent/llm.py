"""Single entry point for every OpenAI call.

- Reads OPENAI_* from .env; never logs the key.
- Structured output via ``responses.parse`` into a pydantic model, then one
  retry on validation failure.
- Per-call timeout and reasoning effort.
- Disk cache keyed by (model, effort, schema, messages): LLM_CACHE_MODE
  off | read | readwrite. Timed in-deadline calls should use cache_mode="off".
- Token accounting with a hard cap (LLM_MAX_TOKENS_TOTAL).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from genco_market.core.config import load_dotenv_once, scratch_dir

T = TypeVar("T", bound=BaseModel)


@dataclass
class CallRecord:
    model: str
    purpose: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cached: bool
    ok: bool
    error: str = ""


@dataclass
class LLMStats:
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.calls if not c.cached)


class LLMBudgetExceeded(RuntimeError):
    pass


class LLM:
    def __init__(self, model: str | None = None, effort: str | None = None, timeout_s: float | None = None):
        load_dotenv_once()
        from openai import OpenAI

        self.model = model or os.environ.get("OPENAI_MODEL_AGENT", "gpt-5.4-mini")
        self.effort = effort or os.environ.get("OPENAI_REASONING_EFFORT", "low")
        self.timeout_s = timeout_s or float(os.environ.get("OPENAI_TIMEOUT_S", "8"))
        self.cache_mode = os.environ.get("LLM_CACHE_MODE", "readwrite")
        self.cache_dir = Path(os.environ.get("LLM_CACHE_DIR", scratch_dir("llm_cache")))
        self.max_tokens_total = int(os.environ.get("LLM_MAX_TOKENS_TOTAL", "5000000"))
        self.client = OpenAI(max_retries=0)
        self.stats = LLMStats()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ cache
    def _key(self, messages: list[dict], schema: type[BaseModel] | None, effort: str) -> str:
        blob = json.dumps(
            {"m": self.model, "e": effort, "s": schema.model_json_schema() if schema else None, "msg": messages},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str, mode: str) -> dict | None:
        if mode == "off":
            return None
        p = self.cache_dir / key[:2] / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _cache_put(self, key: str, value: dict, mode: str) -> None:
        if mode != "readwrite":
            return
        p = self.cache_dir / key[:2] / f"{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    # ------------------------------------------------------------------ calls
    def parse(
        self,
        system: str,
        user: str,
        schema: type[T],
        purpose: str = "",
        timeout_s: float | None = None,
        effort: str | None = None,
        cache_mode: str | None = None,
    ) -> T:
        """Structured call. Raises on timeout/failure; callers own fallbacks."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        effort = effort or self.effort
        mode = cache_mode or self.cache_mode
        key = self._key(messages, schema, effort)
        hit = self._cache_get(key, mode)
        if hit is not None:
            self._record(CallRecord(self.model, purpose, 0.0, 0, 0, cached=True, ok=True))
            return schema.model_validate(hit)
        if self.stats.tokens > self.max_tokens_total:
            raise LLMBudgetExceeded(f"token cap {self.max_tokens_total} reached")

        last_err = ""
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                r = self.client.with_options(timeout=timeout_s or self.timeout_s).responses.parse(
                    model=self.model,
                    input=messages,
                    text_format=schema,
                    reasoning={"effort": effort},
                )
                out = r.output_parsed
                if out is None:
                    raise ValueError("empty parsed output")
                self._record(
                    CallRecord(self.model, purpose, time.perf_counter() - t0,
                               r.usage.input_tokens, r.usage.output_tokens, cached=False, ok=True)
                )
                self._cache_put(key, out.model_dump(mode="json"), mode)
                return out
            except ValidationError as e:  # retry once on schema failure only
                last_err = f"validation: {e}"[:300]
                self._record(CallRecord(self.model, purpose, time.perf_counter() - t0, 0, 0, False, False, last_err))
                continue
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"[:300]
                self._record(CallRecord(self.model, purpose, time.perf_counter() - t0, 0, 0, False, False, last_err))
                break
        raise RuntimeError(f"LLM call failed ({purpose}): {last_err}")

    def _record(self, rec: CallRecord) -> None:
        with self._lock:
            self.stats.calls.append(rec)
