"""Experiment config (YAML -> pydantic) and repo paths."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"


def load_dotenv_once() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env", override=False)


class EpisodeCfg(BaseModel):
    horizon: int = 4  # intervals per episode
    n_episodes: int = 3
    start_hours: list[int] | None = None  # hour-of-year starts; None = sampled


class AgentCfg(BaseModel):
    markups: list[float] = Field(default_factory=lambda: [1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
    deadline_s: float = 20.0
    compute_budget_s: float = 5.0


class SemanticsCfg(BaseModel):
    level: str = "L1"  # L0 | L1 | L2 | L3
    event_types: list[str] = Field(
        default_factory=lambda: ["heatwave", "planned_outage", "thunderstorm", "gas_price"]
    )
    max_events: int = 2
    n_distractors: int | None = None  # None = template default (L2+); 0 = none
    paraphrase: bool = False  # L2/L3: use the cached LLM paraphrases (offline: cache only)


class Config(BaseModel):
    grid: str = "case118_ieee"
    firms_file: str = "firms_case118.yaml"
    seed: int = 0
    episode: EpisodeCfg = Field(default_factory=EpisodeCfg)
    agent: AgentCfg = Field(default_factory=AgentCfg)
    semantics: SemanticsCfg = Field(default_factory=SemanticsCfg)
    solver_workers: int = 8
    oracle_scenarios: int = 200

    @classmethod
    def load(cls, name_or_path: str = "mvp.yaml") -> "Config":
        p = Path(name_or_path)
        if not p.exists():
            p = CONFIGS / name_or_path
        return cls.model_validate(yaml.safe_load(p.read_text()) or {})

    def firms(self) -> dict:
        return yaml.safe_load((CONFIGS / self.firms_file).read_text())


def scratch_dir(*parts: str) -> Path:
    load_dotenv_once()
    base = Path(os.environ.get("GENCO_WORKDIR", REPO / "work"))
    p = base.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
