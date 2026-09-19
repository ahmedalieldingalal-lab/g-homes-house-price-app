"""Phase A, step 20 -- caching on a canonical-input hash. Per
PROJECT_TODO.md: "the real reproducibility guarantee, at zero quota
cost." Two different callers giving the exact same house the exact same
inputs should always see the exact same result (including the exact same
LLM narrative, if one was generated) without a second Gemini call --
otherwise a demo/grading session that re-submits the same house (a very
likely thing to happen) burns quota and risks getting a DIFFERENT
narrative the second time purely from LLM sampling noise, which would
look like a bug even though nothing is actually wrong.

Built as a small, standalone, directly-testable utility rather than
reaching for `st.cache_data` (Streamlit's own decorator) so it can be
verified here, before Step 7's UI exists, and reused as-is once it does.

AUDIT FIX 2026-09-09 (SA29, final audit): this docstring used to also
claim a real Streamlit app "can either wrap `interpret_house` directly
with `@st.cache_data`... or call `SimpleCache.get_or_compute()`... exactly
the same idea" -- that first option doesn't actually work: `interpret_house`
takes an `InterpretationContext` argument, and Streamlit's automatic
argument-hashing refuses non-primitive objects like it
(`UnhashableParamError`), confirmed by actually calling
`st.cache_data(interpret_house)(raw, ctx=ctx)`. `SimpleCache.get_or_compute()`
below is the one that actually works, which is exactly why `app.py` uses it.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Callable


def _non_json_default(v: object) -> dict:
    """AUDIT FIX 2026-09-09 (SA21, final audit): `json.dumps(..., default=str)`
    collapsed any non-JSON-native value to its bare `str()`, so an object
    and a plain string sharing the same text (e.g. `K({'a': 'X'})` and
    `K({'a': obj})` where `str(obj) == 'X'`) produced IDENTICAL cache keys
    -- a real, if latent, cache-poisoning risk (today's only caller,
    `app.py`'s `raw_house`, is all scalars/strings, so this never fires in
    production). Tagging the type name alongside the repr means a plain
    string value and a same-text-but-different-TYPE object no longer
    collide. Two distinct instances of the SAME class with the same
    `str()` still hash identically -- that residual case is an inherent
    property of falling back to a string representation at all (there is
    no stable, run-independent identity to key on instead), not something
    this fix claims to close."""
    return {"__type__": type(v).__qualname__, "__repr__": str(v)}


def canonical_cache_key(raw_input: dict, sale_year: int | None = None, sale_month: int | None = None,
                         gemini_enabled: bool = False) -> str:
    """A deterministic key covering everything that affects
    `interpret_house()`'s output: the raw house fields (order-independent,
    numeric types normalized so `4` and `4.0` hash identically), the
    temporal convention actually used, and whether the Gemini path was
    even attempted (a cached "no key configured" template result must
    never be silently served once a key IS configured).

    AUDIT FIX 2026-09-09 (SA31b, final audit): `sale_year`/`sale_month`
    are never actually passed by this function's one real caller
    (`app.py`, which only ever calls `canonical_cache_key(raw_house,
    gemini_enabled=...)`) -- `interpret_house()` itself does accept both,
    always defaulted to `None` here today. Kept in the signature
    deliberately rather than removed: this key's whole job is to cover
    "everything that affects `interpret_house()`'s output", and if a
    future caller ever DOES vary the sale-date convention per call, a key
    that ignored it would serve one house's cached result for a
    different effective input -- a correctness bug, not just dead
    parameters. Confirmed harmless today only because no caller varies
    them yet."""
    normalized = {}
    for k, v in raw_input.items():
        if isinstance(v, bool):
            normalized[k] = v
        elif isinstance(v, (int, float)):
            normalized[k] = float(v)
        else:
            normalized[k] = v
    payload = {
        "raw_input": normalized,
        "sale_year": sale_year, "sale_month": sale_month,
        "gemini_enabled": gemini_enabled,
    }
    canonical_json = json.dumps(payload, sort_keys=True, default=_non_json_default)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class SimpleCache:
    """A small in-memory LRU cache (no external dependency, no disk I/O --
    appropriate for a single-process demo app per the "no token bucket
    needed, single-user demo app" sizing note for the sibling step 21)."""

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._store: OrderedDict[str, object] = OrderedDict()
        # AUDIT FIX 2026-09-09 (SA31a, final audit): these two counters are
        # read by nothing in app.py or anywhere else this cache is used in
        # production -- confirmed by grep (`\.hits\b|\.misses\b` across the
        # repo matches only this file and the one test below). That is a
        # deliberate choice, not an oversight: they exist as debug/test
        # observability (this project's own SA15 concurrency fix used them
        # directly to prove the race -- "5 threads on one key -> compute_fn
        # called 5 times, hits=0, misses=5" -- and tests/test_step6_cache_
        # and_transport.py's SA15 regression test asserts on them), the same
        # role a hit/miss counter plays in any real cache implementation,
        # even one with no live dashboard reading it yet.
        self.hits = 0
        self.misses = 0
        # AUDIT FIX 2026-09-09 (SA15, final audit): see get_or_compute()'s
        # docstring addition below for why these exist.
        self._store_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_lock = threading.Lock()

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._key_locks_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def get_or_compute(self, key: str, compute_fn: Callable[[], object],
                        should_cache: "Callable[[object], bool] | None" = None) -> object:
        """AUDIT FIX 2026-09-09 (SA2, final audit): added the optional
        `should_cache` predicate. Previously every computed value was
        stored unconditionally, which meant a transient failure baked into
        the cached value (e.g. a Gemini timeout that made
        `interpret_house()` fall back to its template narrative) was
        permanently pinned to this exact input's key -- a user re-
        submitting the identical house after the transient failure passed
        could never get a fresh attempt, because the cache hit fired
        before `compute_fn` ever ran again. `should_cache(value)` lets the
        caller decline to persist a specific result (the value is still
        returned to THIS caller -- only future calls are affected) while
        leaving every other case (the common one) cached exactly as
        before.

        AUDIT FIX 2026-09-09 (SA15, final audit): this module's own
        docstring promises two concurrent identical submissions "see the
        exact same result... without a second Gemini call", but there was
        no locking and no in-flight de-duplication at all -- `app.py`
        shares one `SimpleCache` instance across all Streamlit sessions via
        `@st.cache_resource`, so two concurrent identical submissions used
        to both miss, both call `compute_fn` (a real Gemini call), and race
        on the plain `OrderedDict` mutation (reproduced: 5 threads on one
        key -> `compute_fn` called 5 times, hits=0, misses=5, contradicting
        the docstring outright). Fixed with a per-key lock: only the first
        caller for a given key actually computes; any concurrent caller for
        the SAME key blocks on that key's lock and then re-checks the
        store, so it gets the first caller's result instead of recomputing
        (and, per Streamlit's own single-active-script-run-per-session
        model, this never deadlocks a session against itself). A short
        global `_store_lock` protects the `OrderedDict`'s own mutations
        (insert/move-to-end/evict), which are not otherwise atomic across
        threads. `_key_locks` is intentionally never pruned -- for this
        single-process demo app's `max_size`-bounded key space, that's a
        few hundred small `Lock` objects at most, not an unbounded leak."""
        with self._store_lock:
            if key in self._store:
                self.hits += 1
                self._store.move_to_end(key)
                return self._store[key]

        key_lock = self._get_key_lock(key)
        with key_lock:
            # Re-check: another thread may have finished computing this
            # exact key while we were waiting to acquire its lock.
            with self._store_lock:
                if key in self._store:
                    self.hits += 1
                    self._store.move_to_end(key)
                    return self._store[key]
                self.misses += 1

            value = compute_fn()

            with self._store_lock:
                if should_cache is None or should_cache(value):
                    self._store[key] = value
                    self._store.move_to_end(key)
                    if len(self._store) > self.max_size:
                        self._store.popitem(last=False)
            return value

    def __len__(self) -> int:
        return len(self._store)
