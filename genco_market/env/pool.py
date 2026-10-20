"""Persistent solver pool.

Each worker is a *spawned* process that initializes Julia once (~30 s) with
1 BLAS thread and then serves tasks from a shared queue. Built-in tasks:

    "ac"       clear(scn, markup, info, rules, **kw)  -> ClearingResult
    "dc"       clear_dc(scn, markup, info, rules)     -> ClearingResult (Ipopt)
    "dc_highs" clear_dc(..., backend="highs")         -> ClearingResult (LP)

Any other task name is "package.module:function"; it is called as
fn(*args, **kwargs), with ``info=GridInfo`` added if fn takes ``info``.
A task that raises returns a ``TaskError`` instead of a result; a worker
that dies is replaced and its in-flight task returns a ``TaskError``.
The parent dispatches one task at a time to each worker over its own pipe,
so it always knows which task a dead worker was holding.

Environment clearing and agent rollouts should use *separate* pools so the
agent's compute timing only covers its own tasks.
"""

from __future__ import annotations

import importlib
import inspect
import multiprocessing as mp
import os
from collections import deque
from multiprocessing.connection import wait
import time
import traceback
from dataclasses import dataclass

from genco_market.core.config import Config

_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JULIA_NUM_THREADS": "1",
    "PYTHON_JULIACALL_THREADS": "1",
}


@dataclass
class TaskError:
    task: str
    error: str

    def __bool__(self) -> bool:
        return False


def _builtin(name: str):
    from genco_market.env import market

    if name == "ac":
        return lambda scn, markup, rules=None, *, info, **kw: market.clear(scn, markup, info, rules, **kw)
    if name == "dc":
        return lambda scn, markup, rules=None, *, info, **kw: market.clear_dc(scn, markup, info, rules, **kw)
    if name == "dc_highs":
        return lambda scn, markup, rules=None, *, info, **kw: market.clear_dc(
            scn, markup, info, rules, backend="highs", **kw
        )
    return None


def _resolve(name: str, cache: dict):
    fn = cache.get(name)
    if fn is None:
        fn = _builtin(name)
        if fn is None:
            mod, attr = name.split(":")
            f = getattr(importlib.import_module(mod), attr)
            if "info" in inspect.signature(f).parameters:
                fn = f
            else:
                fn = lambda *a, info, _f=f, **k: _f(*a, **k)  # noqa: E731
        cache[name] = fn
    return fn


def _worker(wid: int, cfg: dict, conn, init_julia: bool) -> None:
    os.environ.update(_THREAD_ENV)
    from genco_market.env.grid import GridInfo

    info = GridInfo(Config.model_validate(cfg))
    if init_julia:
        from genco_market.env.market import get_jl

        get_jl()
    conn.send(("ready", None, None))
    cache: dict = {}
    while True:
        try:
            item = conn.recv()
        except EOFError:
            break
        if item is None:
            break
        job, name, args, kwargs = item
        try:
            out = _resolve(name, cache)(*args, info=info, **kwargs)
        except Exception:
            out = TaskError(name, traceback.format_exc(limit=5))
        conn.send(("done", job, out))


class SolverPool:
    """``with SolverPool(8) as pool: pool.map("ac", [(scn, 1.5, rules), ...])``

    Async use (anytime estimators): ``jobs = [pool.submit("ac", scn, m) ...]``,
    ``pool.collect(jobs, timeout=...)`` returns what finished in time and
    ``pool.cancel_pending()`` drops tasks that have not started.
    """

    def __init__(self, n_workers: int = 8, cfg: Config | None = None, init_julia: bool = True, startup_timeout: float = 600):
        self.n_workers = min(int(n_workers), 16, os.cpu_count() or 1)
        self._cfg = (cfg or Config.load()).model_dump()
        self._init_julia = init_julia
        self._ctx = mp.get_context("spawn")
        self._workers: dict[int, tuple] = {}  # wid -> (process, conn)
        self._ready: set[int] = set()
        self._busy: dict[int, tuple] = {}  # wid -> (job, name)
        self._queue: deque = deque()
        self._done: dict[int, object] = {}
        self._pending: set[int] = set()
        self._next_job = 0
        self._next_wid = 0
        for _ in range(self.n_workers):
            self._spawn()
        t0 = time.perf_counter()
        while len(self._ready) < self.n_workers:
            left = startup_timeout - (time.perf_counter() - t0)
            if left <= 0 or not self._workers:
                self.close()
                raise RuntimeError("solver pool did not start")
            self._pump(min(left, 5.0), respawn=False)
        self.startup_s = time.perf_counter() - t0

    def _spawn(self) -> None:
        wid = self._next_wid
        self._next_wid += 1
        parent, child = self._ctx.Pipe()
        saved = {k: os.environ.get(k) for k in _THREAD_ENV}
        os.environ.update(_THREAD_ENV)  # inherited by the spawned child
        try:
            p = self._ctx.Process(target=_worker, args=(wid, self._cfg, child, self._init_julia), daemon=True)
            p.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        child.close()
        self._workers[wid] = (p, parent)

    def _dispatch(self) -> None:
        for wid in list(self._ready):
            if wid in self._busy:
                continue
            while self._queue and self._queue[0][0] not in self._pending:
                self._queue.popleft()  # cancelled
            if not self._queue:
                return
            item = self._queue.popleft()
            self._busy[wid] = (item[0], item[1])
            self._workers[wid][1].send(item)

    def _finish(self, job: int, out) -> None:
        if job in self._pending:
            self._pending.discard(job)
            self._done[job] = out

    def _pump(self, timeout: float | None, respawn: bool = True) -> None:
        """Wait for messages or dead workers, then hand out queued tasks."""
        by_obj = {}
        for wid, (p, conn) in self._workers.items():
            by_obj[conn] = wid
            by_obj[p.sentinel] = wid
        for obj in wait(list(by_obj), timeout=timeout):
            wid = by_obj[obj]
            if wid not in self._workers:
                continue
            p, conn = self._workers[wid]
            try:
                if obj is conn:
                    kind, job, out = conn.recv()
                    if kind == "ready":
                        self._ready.add(wid)
                    else:
                        self._busy.pop(wid, None)
                        self._finish(job, out)
                    continue
            except (EOFError, OSError):
                pass
            if p.is_alive():
                continue
            # worker died (crash inside a solve, or at start-up)
            del self._workers[wid]
            self._ready.discard(wid)
            job, name = self._busy.pop(wid, (None, "?"))
            if job is not None:
                self._finish(job, TaskError(name, f"worker died (exit {p.exitcode})"))
            if respawn:
                self._spawn()  # the replacement pays the Julia start-up again
        self._dispatch()

    # ------------------------------------------------------------------
    def submit(self, name: str, *args, **kwargs) -> int:
        job = self._next_job
        self._next_job += 1
        self._pending.add(job)
        self._queue.append((job, name, args, kwargs))
        self._dispatch()
        return job

    def collect(self, jobs, timeout: float | None = None) -> dict[int, object]:
        """Results of ``jobs`` that finish within ``timeout`` (None = all)."""
        jobs = list(jobs)
        t0 = time.perf_counter()
        while any(j not in self._done and j in self._pending for j in jobs):
            left = None if timeout is None else timeout - (time.perf_counter() - t0)
            if left is not None and left <= 0:
                break
            self._pump(5.0 if left is None else min(left, 5.0))
        return {j: self._done.pop(j) for j in jobs if j in self._done}

    def map(self, name: str, arg_list, timeout: float | None = None) -> list:
        """Run ``name`` on each args tuple; ordered results (None if not done by timeout)."""
        jobs = [self.submit(name, *(a if isinstance(a, tuple) else (a,))) for a in arg_list]
        got = self.collect(jobs, timeout)
        return [got.get(j) for j in jobs]

    def cancel(self, jobs) -> None:
        """Forget ``jobs``: queued ones never start, running ones' results are
        dropped on arrival, finished-but-uncollected ones are discarded."""
        for j in jobs:
            self._pending.discard(j)
            self._done.pop(j, None)

    def cancel_pending(self) -> int:
        """Drop queued (not yet started) tasks; returns how many were dropped.
        Results of tasks already running are discarded when they arrive."""
        n = len(self._queue)
        for item in self._queue:
            self._pending.discard(item[0])
        self._queue.clear()
        return n

    def close(self) -> None:
        for p, conn in self._workers.values():
            try:
                conn.send(None)
            except (OSError, BrokenPipeError):
                pass
        for p, conn in self._workers.values():
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
            conn.close()
        self._workers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
