"""Deterministic seed streams for every experiment and cell.

Convention (frozen in formalization_memo.md Section 1):
    stream(master_seed, "experiment", "cell_id", ...) -> Generator

Every result row must record (experiment, cell_id, seed) so any number can be
regenerated bit-for-bit. Master seed lives in Research/seeds.yaml.
"""
from __future__ import annotations

import hashlib
import numpy as np

MASTER_SEED = 20260823


def stream(master_seed: int, *path) -> np.random.Generator:
    """Deterministic generator from a master seed and a tuple of string keys."""
    ss = np.random.SeedSequence(entropy=master_seed)
    if path:
        keys = [int.from_bytes(hashlib.sha256(str(p).encode()).digest()[:4], "big") for p in path]
        ss = np.random.SeedSequence(entropy=master_seed, spawn_key=tuple(keys))
    return np.random.default_rng(ss)


def cell_stream(experiment: str, cell_id: str, rep_index: int | None = None,
                master_seed: int = MASTER_SEED) -> np.random.Generator:
    """Per-cell (optionally per-replication) stream."""
    path = [experiment, cell_id] + ([f"rep{rep_index}"] if rep_index is not None else [])
    return stream(master_seed, *path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
