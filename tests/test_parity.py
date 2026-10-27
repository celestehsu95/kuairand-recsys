"""Training/serving parity: the Redis-backed store must produce exactly the features the
offline backfill (in-memory store) produces for the same event sequence.

    ./demo.sh test          # runs inside the stack against the real Redis
"""
import os
import random
import sys

import numpy as np
import redis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realtime_features import FeatureLogic, InMemoryStore, RedisStore  # noqa: E402


def main():
    r = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), port=6379, db=15)  # separate db
    r.flushdb()
    rng = random.Random(0)
    users, items = range(1, 31), range(1, 51)
    events = [(rng.randint(1, 30), rng.randint(1, 50), rng.random() < 0.3) for _ in range(5000)]

    mem, red = InMemoryStore(), RedisStore(r, prefix="test")
    for e in events[:2500]:
        mem.record(*e)
        red.record(*e)
    mem.record_many(events[2500:])
    red.record_many(events[2500:])
    assert all(mem.get_user(u) == red.get_user(u) for u in users), "user state differs"
    assert mem.get_items(items) == red.get_items(items), "item state differs"
    for a, b in zip(mem.get_item_stats(items), red.get_item_stats(items)):
        assert np.array_equal(a, b), "item aggregates differ"

    r.flushdb()
    red.load_from(mem)
    assert all(mem.get_user(u) == red.get_user(u) for u in users), "bootstrap load differs"
    for a, b in zip(mem.get_item_stats(items), red.get_item_stats(items)):
        assert np.array_equal(a, b), "bootstrap item aggregates differ"
    more = [(rng.randint(1, 30), rng.randint(1, 50), rng.random() < 0.3) for _ in range(3000)]
    mem.record_many(more)
    red.record_many(more)  # aggregates must stay exact when streaming continues after bootstrap
    for a, b in zip(mem.get_item_stats(items), red.get_item_stats(items)):
        assert np.array_equal(a, b), "item aggregates drift after bootstrap"

    emb = np.random.default_rng(0).normal(size=(51, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    logic = FeatureLogic(emb, np.arange(51) % 5, np.arange(51) % 7, 0.3)
    for u in users:
        assert np.allclose(logic.compute(mem, u, list(items)), logic.compute(red, u, list(items))), f"features differ for user {u}"
    r.flushdb()
    print("Parity test passed: online (Redis) features == offline (backfill) features")


if __name__ == "__main__":
    main()
