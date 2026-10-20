"""Deterministic, named random streams: rng(seed, "world", ep, t) is stable."""

from __future__ import annotations

import hashlib

import numpy as np


def rng(*keys) -> np.random.Generator:
    h = hashlib.sha256("/".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def sample_start_hour(seed: int, ep: int) -> int:
    """Episode start: a random day of 2024 (ERCOT files cover 8,268 h), 08:00-18:00."""
    r = rng(seed, "start_hour", ep)
    return int(r.integers(0, 8268 // 24 - 1)) * 24 + int(r.integers(8, 19))
