"""Stream processor: Kafka "events" -> Redis feature store.

Messages are keyed by user id, so all of a user's events land on one partition and are
applied in order. Each poll batch is written to Redis in a single pipelined round trip.
End-to-end lag (event produced -> features updated) is tracked for the demo and README.
"""
import json
import os
import time

import redis
from confluent_kafka import Consumer, KafkaError

from realtime_features import RedisStore

KAFKA = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.environ.get("EVENTS_TOPIC", "events")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")


def main():
    r = redis.Redis(host=REDIS_HOST, port=6379)
    store = RedisStore(r)
    consumer = Consumer({
        "bootstrap.servers": KAFKA,
        "group.id": "feature-worker",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([TOPIC])
    print(f"feature-worker consuming {TOPIC} from {KAFKA}", flush=True)
    lag_ema, total, last_log = None, 0, time.time()
    while True:
        # consume() waits for the full timeout unless the batch fills up, so keep it short for low lag
        msgs = consumer.consume(num_messages=500, timeout=0.05)
        batch, lags = [], []
        for m in msgs:
            if m.error():
                if m.error().code() not in (KafkaError._PARTITION_EOF, KafkaError.UNKNOWN_TOPIC_OR_PART):
                    print(f"Kafka error: {m.error()}", flush=True)
                continue
            ev = json.loads(m.value())
            batch.append((int(ev["uid"]), int(ev["iid"]), int(ev["clicked"])))
            if "sent_ms" in ev:
                lags.append(time.time() * 1000 - ev["sent_ms"])
        if not batch:
            continue
        store.record_many(batch)  # same record() semantics the offline backfill used
        total += len(batch)
        pipe = r.pipeline(transaction=False)
        pipe.incrby("rt:stats:events", len(batch))
        if lags:
            last = lags[-1]
            for x in lags:
                lag_ema = x if lag_ema is None else 0.95 * lag_ema + 0.05 * x
            pipe.set("rt:stats:last_lag_ms", round(last, 2))
            pipe.set("rt:stats:lag_ema_ms", round(lag_ema, 2))
        pipe.execute()
        if time.time() - last_log > 10:
            print(f"processed {total:,} events, lag_ema {lag_ema:.1f} ms" if lag_ema else f"processed {total:,}", flush=True)
            last_log = time.time()


if __name__ == "__main__":
    main()
