"""Versioned, deterministic quotas; preserves the original detent RNG draw."""
import random
from functools import lru_cache
VERSION = "force_direction_balance_20260909_v1"
def original_sign(seed):
    rng = random.Random(seed)
    rng.random()
    return rng.choice((-1, 1))
@lru_cache(None)
def mapping():
    result = {}
    groups = [list(range(p, 200, 2)) for p in (0, 1)] + [list(range(a, a+100)) for a in (200, 300)]
    for seeds in groups:
        signs = {seed: original_sign(seed) for seed in seeds}
        excess = sum(v == -1 for v in signs.values()) - len(seeds)//2
        majority = -1 if excess > 0 else 1
        candidates = [s for s in seeds if signs[s] == majority]
        # Minimal flips; fixed ascending seed ordering, never outcome selection.
        for seed in candidates[:abs(excess)]:
            signs[seed] *= -1
        result.update(signs)
    return result
def balanced_sign(seed, original):
    return mapping().get(int(seed), original)
def final_move_sign(seed):
    if not 0 <= seed < 200:
        return -1
    peers = [s for s in range(seed % 2, 200, 2) if mapping()[s] == mapping()[seed]]
    return 1 if peers.index(seed) % 2 == 0 else -1
