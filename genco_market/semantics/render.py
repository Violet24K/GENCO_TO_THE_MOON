"""Template rendering. Pure functions: the caller supplies the
rng, so all wording choices come from named seeding streams."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from genco_market.core.config import CONFIGS
from genco_market.core.schemas import PhysicalState

TEMPLATE_DIR = CONFIGS / "templates"
YEAR0 = datetime(2024, 1, 1)  # start_hour is hour-of-year in 2024 (ERCOT profile year)


@lru_cache(maxsize=4)
def _load_dir(path: str) -> dict:
    return {p.stem: yaml.safe_load(p.read_text()) for p in sorted(Path(path).glob("*.yaml"))}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_templates(path: Path = TEMPLATE_DIR) -> dict:
    return _load_dir(str(path))


def clock(start_hour: int, t: int) -> datetime:
    return YEAR0 + timedelta(hours=start_hour + t)


def fmt_window(common: dict, start_hour: int, t0: int, t1: int) -> str:
    """Interval t covers [hour, hour+1); the window [t0, t1] is inclusive."""
    a, b = clock(start_hour, t0), clock(start_hour, t1 + 1)
    day = a.strftime("%a %b %d")
    end = b.strftime("%H:%M") if b.date() == a.date() or b.hour == 0 else b.strftime("%H:%M on %a %b %d")
    fmt = common["window_format_single"] if t0 == t1 else common["window_format"]
    return fmt.format(start=a.strftime("%H:%M"), end=end, day=day, t0=t0, t1=t1)


def fmt_since(common: dict, start_hour: int, t0: int) -> str:
    a = clock(start_hour, t0)
    return common["since_format"].format(start=a.strftime("%H:%M"), day=a.strftime("%a %b %d"), t0=t0)


def zone_slots(common: dict, z: str | None) -> dict:
    if z is None:
        return {"zone": common["system_name"], "zone_short": "the system", "region": "the whole system"}
    zd = common["zones"][z]
    return {"zone": zd["name"], "zone_short": zd["short"], "region": zd["region"]}


def pick_phrase(table: list[dict], p: float, r: np.random.Generator) -> tuple[str, list[float]]:
    """Hedge phrase whose hidden range [lo, hi) contains p."""
    for row in table:
        lo, hi = row["range"]
        if lo <= p < hi:
            return str(r.choice(row["phrases"])), [lo, hi]
    raise ValueError(f"no hedge for p={p}")


def choose_template(group: dict, source: str, r: np.random.Generator, key: str) -> tuple[str, str]:
    """Returns (template_text, template_key) for one source in a template group."""
    opts = group[source]
    i = int(r.integers(len(opts)))
    return opts[i], f"{key}/{source}/{i}"


def fill(template: str, slots: dict) -> str:
    text = template.format_map(slots)
    return text[0].upper() + text[1:] if template.startswith("{") else text


def render_l0(state: PhysicalState) -> str:
    """Debug level: the true PhysicalState as JSON."""
    return json.dumps(state.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
