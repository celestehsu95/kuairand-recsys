"""Real-time features, defined ONCE and used by both offline training and online serving.

Training/serving skew is one of the most common ways recommenders silently break: the
features a model sees in production are computed by different code than the features it
was trained on. Here both paths call the same `FeatureLogic`; only the storage differs:

    offline backfill  -> InMemoryStore (dicts of deques, replayed in event-time order)
    online serving    -> RedisStore    (Redis lists, updated by the Kafka consumer)

All windows are count-based (last N events), not wall-clock-based, so replaying a log at
any speed yields the same feature values as the original timeline.
"""
from collections import defaultdict, deque

import numpy as np

USER_CLICKS_N = 20       # last N clicked items per user
USER_EXPOSURES_N = 50    # last N exposures per user (click / no click)
ITEM_EXPOSURES_N = 200   # last N exposures per item

FEATURE_NAMES = [
    "user_recent_clicks",     # how much short-term history we have (0..1)
    "sim_to_recent_clicks",   # cosine(candidate, mean of recently clicked item embeddings)
    "tag_match_recent",       # share of recent clicks with the candidate's tag
    "author_match_recent",    # share of recent clicks from the candidate's author
    "user_recent_ctr",        # user's smoothed CTR over last exposures
    "item_recent_ctr",        # item's smoothed CTR over last exposures
    "item_recent_volume",     # how many recent exposures the item has (0..1)
]
N_FEATURES = len(FEATURE_NAMES)


# ============================ storage ============================
class InMemoryStore:
    """Newest-first lists, same ordering as Redis LPUSH."""

    def __init__(self):
        self.user_clicks = defaultdict(lambda: deque(maxlen=USER_CLICKS_N))
        self.user_exp = defaultdict(lambda: deque(maxlen=USER_EXPOSURES_N))
        self.item_exp = defaultdict(lambda: deque(maxlen=ITEM_EXPOSURES_N))

    def record(self, uid, iid, clicked):
        self.user_exp[uid].appendleft(int(clicked))
        self.item_exp[iid].appendleft(int(clicked))
        if clicked:
            self.user_clicks[uid].appendleft(int(iid))

    def record_many(self, events):
        for uid, iid, clicked in events:
            self.record(uid, iid, clicked)

    def get_user(self, uid):
        return list(self.user_clicks.get(uid, ())), list(self.user_exp.get(uid, ()))

    def get_items(self, iids):
        return [list(self.item_exp.get(int(i), ())) for i in iids]

    def get_item_stats(self, iids):
        """(clicks, exposures) over each item's window."""
        hist = [self.item_exp.get(int(i), ()) for i in iids]
        return np.array([sum(h) for h in hist], np.float64), np.array([len(h) for h in hist], np.float64)


# Item windows are read for every candidate on every request, so the online store keeps their
# aggregates up to date on write instead of returning full lists on read. The script pushes the
# new outcome, evicts the oldest one beyond the window, and adjusts the running click count
# atomically, so it stays exact with any number of workers.
ITEM_WINDOW_LUA = """
local n = redis.call('LPUSH', KEYS[1], ARGV[1])
local maxlen = tonumber(ARGV[2])
local delta = tonumber(ARGV[1])
if n > maxlen then
  delta = delta - tonumber(redis.call('RPOP', KEYS[1]))
  n = maxlen
end
if delta ~= 0 then redis.call('HINCRBY', KEYS[2], ARGV[3] .. ':s', delta) end
redis.call('HSET', KEYS[2], ARGV[3] .. ':n', n)
return n
"""


class RedisStore:
    """Same semantics as InMemoryStore, backed by Redis.

    user windows: lists (LPUSH + LTRIM), read in full (at most 70 values per request)
    item windows: list + running (clicks, count) in one hash, maintained by ITEM_WINDOW_LUA;
                  a request reads 2 numbers per candidate with a single HMGET
    """

    def __init__(self, client, prefix="rt"):
        self.r = client
        self.p = prefix

    def _k(self, kind, key):
        return f"{self.p}:{kind}:{key}"

    @property
    def _istat(self):
        return f"{self.p}:istat"

    def _record(self, pipe, uid, iid, clicked):
        self._push(pipe, self._k("uexp", uid), int(clicked), USER_EXPOSURES_N)
        pipe.eval(ITEM_WINDOW_LUA, 2, self._k("iexp", iid), self._istat, int(clicked), ITEM_EXPOSURES_N, int(iid))
        if clicked:
            self._push(pipe, self._k("uclk", uid), int(iid), USER_CLICKS_N)

    def record(self, uid, iid, clicked):
        pipe = self.r.pipeline(transaction=False)
        self._record(pipe, uid, iid, clicked)
        pipe.execute()

    @staticmethod
    def _push(pipe, key, value, maxlen):
        pipe.lpush(key, value)
        pipe.ltrim(key, 0, maxlen - 1)

    def record_many(self, events):
        """events: iterable of (uid, iid, clicked). One round trip for the whole batch."""
        pipe = self.r.pipeline(transaction=False)
        for uid, iid, clicked in events:
            self._record(pipe, uid, iid, clicked)
        pipe.execute()

    def get_user(self, uid):
        pipe = self.r.pipeline(transaction=False)
        pipe.lrange(self._k("uclk", uid), 0, -1)
        pipe.lrange(self._k("uexp", uid), 0, -1)
        clicks, exp = pipe.execute()
        return [int(x) for x in clicks], [int(x) for x in exp]

    def get_items(self, iids):
        pipe = self.r.pipeline(transaction=False)
        for i in iids:
            pipe.lrange(self._k("iexp", int(i)), 0, -1)
        return [[int(x) for x in v] for v in pipe.execute()]

    def get_item_stats(self, iids):
        fields = []
        for i in iids:
            fields += [f"{int(i)}:s", f"{int(i)}:n"]
        vals = self.r.hmget(self._istat, fields)
        v = np.array([float(x) if x is not None else 0.0 for x in vals])
        return v[0::2], v[1::2]

    def load_from(self, mem, batch=2000):
        """Bulk-load a backfilled InMemoryStore into Redis (feature-store bootstrap)."""
        pipe, n = self.r.pipeline(transaction=False), 0
        pipe.delete(self._istat)
        for kind, table in [("uclk", mem.user_clicks), ("uexp", mem.user_exp), ("iexp", mem.item_exp)]:
            for key, values in table.items():
                k = self._k(kind, key)
                pipe.delete(k)
                if values:
                    pipe.rpush(k, *values)  # values are newest-first; RPUSH keeps that order
                if kind == "iexp":
                    pipe.hset(self._istat, f"{key}:s", int(sum(values)))
                    pipe.hset(self._istat, f"{key}:n", len(values))
                n += 1
                if n % batch == 0:
                    pipe.execute()
        pipe.execute()
        return n


# ============================ features ============================
class FeatureLogic:
    def __init__(self, item_emb, item_tag, item_author, global_ctr):
        """item_emb: [n_items+1, d] L2-normalized (row 0 = padding); item_tag/item_author: [n_items+1] codes."""
        self.item_emb = np.asarray(item_emb, dtype=np.float32)
        self.item_tag = np.asarray(item_tag)
        self.item_author = np.asarray(item_author)
        self.g = float(global_ctr)

    def update(self, store, uid, iid, clicked):
        store.record(int(uid), int(iid), int(clicked))

    def recent_click_vector(self, clicks):
        if not clicks:
            return None
        m = self.item_emb[clicks].mean(0)
        n = np.linalg.norm(m)
        return m / n if n > 0 else None

    def compute(self, store, uid, iids, user_state=None):
        """Features for one user and a batch of candidate items -> float32 [len(iids), N_FEATURES].
        Pass `user_state` (from store.get_user) to avoid fetching it twice."""
        iids = np.asarray(iids, dtype=np.int64)
        clicks, exp = user_state if user_state is not None else store.get_user(int(uid))
        out = np.zeros((len(iids), N_FEATURES), dtype=np.float32)
        out[:, 0] = len(clicks) / USER_CLICKS_N
        m = self.recent_click_vector(clicks)
        if m is not None:
            out[:, 1] = self.item_emb[iids] @ m
            c = np.asarray(clicks)
            out[:, 2] = (self.item_tag[c][None, :] == self.item_tag[iids][:, None]).mean(1)
            out[:, 3] = (self.item_author[c][None, :] == self.item_author[iids][:, None]).mean(1)
        out[:, 4] = (sum(exp) + 5 * self.g) / (len(exp) + 5)
        clicks_i, n_i = store.get_item_stats(iids)
        out[:, 5] = (clicks_i + 20 * self.g) / (n_i + 20)
        out[:, 6] = n_i / ITEM_EXPOSURES_N
        return out
