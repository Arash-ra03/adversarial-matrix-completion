"""Seed handling that keeps hidden instances unrecoverable by agents.

Two *different* seeds matter, and conflating them breaks the competition:

* The **generation seed** builds ``(X, Z, Y_gt)``. It is private to the grader
  and must never reach an agent. ``generate_instance`` is public and
  deterministic, so anyone holding the generation seed — or able to brute-force
  it from the fully-visible ``X`` — can rebuild the ground truth exactly.

* The **agent seed** is handed to ``solve``/``attack`` purely so an agent's own
  RNG can be deterministic (the robustness check rewards reproducibility). It is
  derived from the generation seed by a one-way function and reveals nothing
  about it.

Modes
-----
* **Transparent (default).** ``resolve_generation_seed`` is the identity, so the
  *public* grader is fully reproducible for self-testing. The agent seed is
  still decoupled, so the naive ``generate_instance(seed, budget)`` cheat fails
  even here; only same-process / brute-force tricks remain, and those only fool
  the participant's own local run.

* **Hidden (graded).** The hidden grader sets ``MATRIX_ARENA_SEED_KEY`` (and
  optionally ``MATRIX_ARENA_SALT``) to secrets. Generation seeds then become
  64-bit keyed hashes — high-entropy and infeasible to brute-force from the
  visible features — and the agent seed becomes non-invertible. This defeats
  both the seed-reconstruction and the X-matching attacks.

The hidden grader should *also* run each agent in an isolated subprocess so a
malicious agent cannot read ``Y_gt`` out of the grader's process memory; seed
hygiene and process isolation are complementary, not substitutes.
"""
from __future__ import annotations

import hashlib
import os

DEFAULT_SALT = "matrix-arena/public"


def _digest_int(*parts: object, nbytes: int = 4) -> int:
    """Deterministic non-negative int from a SHA-256 of the joined parts."""
    payload = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:nbytes], "big")


def resolve_generation_seed(index: int) -> int:
    """Map a public instance index to the actual (possibly secret) generation seed.

    Identity by default (transparent self-testing). If ``MATRIX_ARENA_SEED_KEY``
    is set, returns a 64-bit keyed hash so the generation seed is high-entropy
    and cannot be recovered by brute-forcing which seed reproduces the visible X.
    """
    key = os.environ.get("MATRIX_ARENA_SEED_KEY")
    if key is None:
        return int(index)
    return _digest_int(key, "gen", index, nbytes=8)


def derive_agent_seed(generation_seed: int, role: str = "solve") -> int:
    """One-way per-call seed handed to an agent, decoupled from the generation seed.

    Deterministic (so reproducibility checks still work) but not usable to
    rebuild the instance. With a secret ``MATRIX_ARENA_SALT`` it is also
    non-invertible even when the generation seed space is small.
    """
    salt = os.environ.get("MATRIX_ARENA_SALT", DEFAULT_SALT)
    return _digest_int(salt, role, generation_seed, nbytes=4)
