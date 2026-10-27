"""FastAPI service: recommendations, event ingestion, demo page.

GET  /recommend?uid=123&k=20&candidates=200&realtime=true
POST /event        {"uid": 123, "iid": 456, "clicked": true}   -> Kafka topic "events"
GET  /stats        stream-processing stats written by the worker
GET  /             interactive demo
"""
import json
import os
import time
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from realtime_features import RedisStore
from serving.engine import Engine

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
KAFKA = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.environ.get("EVENTS_TOPIC", "events")
STATIC = os.path.join(os.path.dirname(__file__), "static")

state = {}


@asynccontextmanager
async def lifespan(app):
    r = redis.Redis(host=REDIS_HOST, port=6379)
    state["redis"] = r
    state["engine"] = Engine(ART_DIR, RedisStore(r))
    try:
        from confluent_kafka import Producer
        state["producer"] = Producer({"bootstrap.servers": KAFKA, "linger.ms": 5})
    except Exception as e:  # recommendations still work without Kafka
        print(f"Kafka producer unavailable: {e}")
    yield
    if "producer" in state:
        state["producer"].flush(5)


app = FastAPI(title="KuaiRand two-stage recommender", lifespan=lifespan)


class Event(BaseModel):
    uid: int
    iid: int
    clicked: bool = True


@app.get("/health")
def health():
    e = state["engine"]
    return {"ok": True, "n_users": e.meta["n_users"], "n_items": e.meta["n_items"], "index": e.index_type}


@app.get("/recommend")
def recommend(uid: int, k: int = 20, candidates: int = 200, realtime: bool = True):
    try:
        return state["engine"].recommend(uid, k=k, n_candidates=candidates, realtime=realtime)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/event")
def event(ev: Event):
    p = state.get("producer")
    if p is None:
        raise HTTPException(503, "Kafka producer not available")
    payload = {"uid": ev.uid, "iid": ev.iid, "clicked": int(ev.clicked), "sent_ms": time.time() * 1000, "source": "demo"}
    p.produce(TOPIC, key=str(ev.uid), value=json.dumps(payload))
    p.poll(0)
    return {"ok": True}


@app.get("/demo/users")
def demo_users():
    return {"users": state["engine"].demo_users}


@app.get("/stats")
def stats():
    r = state["redis"]
    vals = r.mget("rt:stats:events", "rt:stats:last_lag_ms", "rt:stats:lag_ema_ms", "rt:stats:bootstrap_events")
    f = lambda v: float(v) if v is not None else None
    return {"events_processed": f(vals[0]), "last_lag_ms": f(vals[1]), "lag_ema_ms": f(vals[2]), "bootstrapped_events": f(vals[3])}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
