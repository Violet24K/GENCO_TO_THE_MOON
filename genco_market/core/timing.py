"""Wall-clock deadline helper for anytime computations."""

from __future__ import annotations

import time


class Deadline:
    def __init__(self, seconds: float):
        self.start = time.perf_counter()
        self.seconds = seconds

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def remaining(self) -> float:
        return max(0.0, self.seconds - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0.0
