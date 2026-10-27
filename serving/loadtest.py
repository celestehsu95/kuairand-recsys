"""Closed-loop load test: N concurrent clients hammer /recommend for a fixed duration.

Reports throughput, client-side p50/p95/p99, and the server's per-stage breakdown.

    python -m serving.loadtest --url http://api:8000 --concurrency 16 --duration 30
"""
import argparse
import asyncio
import json
import os
import random
import time

import httpx
import numpy as np


async def client(http, url, uids, args, deadline, lat, stages, errors):
    while time.perf_counter() < deadline:
        uid = random.choice(uids)
        t = time.perf_counter()
        try:
            r = await http.get(f"{url}/recommend", params={"uid": uid, "k": 20, "candidates": args.candidates,
                                                          "realtime": str(not args.no_realtime).lower()})
            r.raise_for_status()
            lat.append((time.perf_counter() - t) * 1000)
            stages.append(r.json()["timings_ms"])
        except Exception:
            errors.append(1)


async def run(args):
    async with httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=args.concurrency)) as http:
        n_users = (await http.get(f"{args.url}/health")).json()["n_users"]
        uids = list(range(1, n_users + 1))
        for _ in range(20):  # warm-up
            await http.get(f"{args.url}/recommend", params={"uid": random.choice(uids)})
        lat, stages, errors = [], [], []
        start = time.perf_counter()
        deadline = start + args.duration
        await asyncio.gather(*[client(http, args.url, uids, args, deadline, lat, stages, errors) for _ in range(args.concurrency)])
        elapsed = time.perf_counter() - start

    lat = np.array(lat)
    res = {
        "concurrency": args.concurrency, "candidates": args.candidates, "realtime": not args.no_realtime,
        "requests": int(len(lat)), "errors": len(errors), "qps": round(len(lat) / elapsed, 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "server_stage_mean_ms": {k: round(float(np.mean([s[k] for s in stages])), 2) for k in stages[0]},
    }
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        existing = json.load(open(args.out)) if os.path.exists(args.out) else []
        json.dump(existing + [res], open(args.out, "w"), indent=2)
        print(f"Appended to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--no_realtime", action="store_true")
    ap.add_argument("--out", default="results/latency.json")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
