"""Replays the test-period log into Kafka as live traffic.

    python -m serving.replayer --rate 300      # events per second
"""
import argparse
import json
import os
import time

import numpy as np
from confluent_kafka import Producer

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
KAFKA = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.environ.get("EVENTS_TOPIC", "events")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=float(os.environ.get("REPLAY_RATE", "300")))
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    ev = np.load(os.path.join(ART_DIR, "events.npz"))
    test = ev["split"] == "test"
    uids, iids, clicks = ev["uid"][test], ev["iid"][test], ev["is_click"][test]
    p = Producer({"bootstrap.servers": KAFKA, "linger.ms": 10})
    print(f"Replaying {len(uids):,} test-period events at {args.rate:.0f}/s", flush=True)

    batch = max(1, int(args.rate / 20))  # send in 50ms slices
    while True:
        start = time.time()
        for s in range(0, len(uids), batch):
            now = time.time() * 1000
            for u, i, c in zip(uids[s:s + batch], iids[s:s + batch], clicks[s:s + batch]):
                p.produce(TOPIC, key=str(int(u)), value=json.dumps(
                    {"uid": int(u), "iid": int(i), "clicked": int(c), "sent_ms": now, "source": "replay"}))
            p.poll(0)
            sent = s + batch
            ahead = sent / args.rate - (time.time() - start)
            if ahead > 0:
                time.sleep(ahead)
            if sent % (batch * 200) == 0:
                print(f"  sent {sent:,}/{len(uids):,}", flush=True)
        p.flush(10)
        if not args.loop:
            break
    print("Replay finished.", flush=True)


if __name__ == "__main__":
    main()
