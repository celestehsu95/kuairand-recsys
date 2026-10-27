"""Feature-store bootstrap: rebuild online state from history, then hand off to streaming.

Replays the train+val period through the same record() logic as the offline backfill (in
memory, fast), bulk-loads the result into Redis, and creates the Kafka topic. After this the
store holds exactly the state the model would have seen at the start of the test period, and
the worker keeps it current from the stream.
"""
import os
import time

import numpy as np
import redis

from realtime_features import InMemoryStore, RedisStore

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
KAFKA = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.environ.get("EVENTS_TOPIC", "events")


def create_topic():
    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient({"bootstrap.servers": KAFKA})
    futures = admin.create_topics([NewTopic(TOPIC, num_partitions=4, replication_factor=1)])
    for t, f in futures.items():
        try:
            f.result()
            print(f"Created Kafka topic '{t}'")
        except Exception as e:
            print(f"Kafka topic '{t}': {e}")


def main():
    t0 = time.time()
    ev = np.load(os.path.join(ART_DIR, "events.npz"))
    hist = ev["split"] != "test"
    mem = InMemoryStore()
    mem.record_many(zip(ev["uid"][hist].tolist(), ev["iid"][hist].tolist(), ev["is_click"][hist].tolist()))
    print(f"Replayed {hist.sum():,} historical events in memory ({time.time() - t0:.1f}s)")

    r = redis.Redis(host=REDIS_HOST, port=6379)
    r.flushdb()
    n = RedisStore(r).load_from(mem)
    r.set("rt:stats:bootstrap_events", int(hist.sum()))
    r.set("rt:stats:events", 0)
    print(f"Loaded {n:,} keys into Redis ({time.time() - t0:.1f}s)")
    create_topic()


if __name__ == "__main__":
    main()
