"""The two-stage recommendation path, independent of the web framework.

    request -> user vector (+ real-time blend) -> ANN retrieval -> features -> ranker -> top-k

Each stage is timed so the API can report a latency breakdown.
"""
import csv
import json
import os
import time
from datetime import datetime

import numpy as np

from realtime_features import FeatureLogic

try:
    import faiss
except ImportError:  # local testing without faiss: exact search in numpy
    faiss = None


class Ranker:
    """Loads a ranker exported by train_rt_ranker.py and scores (X, dense) on CPU."""

    def __init__(self, art_dir, name):
        import torch

        from models import DCNv2

        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
        with open(os.path.join(art_dir, f"ranker_{name}.json")) as f:
            cfg = json.load(f)
        self.torch = torch
        self.model = DCNv2(cfg["vocabs"], emb_dim=cfg["emb_dim"], n_cross=cfg["n_cross"],
                           use_cross=cfg["use_cross"], n_dense=cfg["n_dense"])
        self.model.load_state_dict(torch.load(os.path.join(art_dir, f"ranker_{name}.pt"), map_location="cpu"))
        self.model.eval()
        self.uses_dense = cfg["n_dense"] > 0
        self.cols = cfg.get("rt_feature_idx")  # which real-time features this variant was trained on

    def score(self, X, dense):
        with self.torch.inference_mode():
            x = self.torch.from_numpy(X)
            if self.uses_dense and self.cols is not None:
                dense = np.ascontiguousarray(dense[:, self.cols])
            d = self.torch.from_numpy(dense) if self.uses_dense else None
            return self.torch.sigmoid(self.model(x, d)).numpy()


class Engine:
    def __init__(self, art_dir, store, rankers=None, index_type=None, rt_alpha=None):
        self.store = store
        with open(os.path.join(art_dir, "meta.json")) as f:
            self.meta = json.load(f)
        feats = np.load(os.path.join(art_dir, "features.npz"))
        self.user_feats, self.item_feats = feats["user_feats"], feats["item_feats"]
        emb = np.load(os.path.join(art_dir, "emb.npz"))
        self.user_emb, self.item_emb = emb["user_emb"], emb["item_emb"]
        self.n_items = self.meta["n_items"]
        with open(os.path.join(art_dir, "demo_users.json")) as f:
            self.demo_users = json.load(f)
        cfg_path = os.path.join(art_dir, "serving_config.json")
        self.default_tab = json.load(open(cfg_path))["default_tab"] if os.path.exists(cfg_path) else 0
        self.rt_alpha = float(rt_alpha if rt_alpha is not None else os.environ.get("RT_ALPHA", "1.0"))

        names = self.meta["item_feat_names"]
        col = lambda n: self.item_feats[:, names.index(n)] if n in names else np.zeros(len(self.item_feats), np.int64)
        self.logic = FeatureLogic(self.item_emb, col("tag_first"), col("author_id"), self.meta["overall_ctr"])

        self.items_meta = {}
        with open(os.path.join(art_dir, "items_meta.csv")) as f:
            for row in csv.DictReader(f):
                self.items_meta[int(row["iid"])] = row

        # ANN index over items 1..n (row r <-> item id r+1)
        self.index_type = index_type or os.environ.get("FAISS_INDEX", "flat")
        vecs = np.ascontiguousarray(self.item_emb[1:], dtype=np.float32)
        if faiss is None:
            self.index = None
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(vecs.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = int(os.environ.get("HNSW_EF", "128"))
            self.index.add(vecs)
        else:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
        self.rankers = rankers if rankers is not None else {"rt": Ranker(art_dir, "rt"), "base": Ranker(art_dir, "base")}

    # ---------------- stages ----------------
    def _retrieve(self, u, n):
        q = u.reshape(1, -1).astype(np.float32)
        if self.index is not None:
            scores, idx = self.index.search(q, n)
            return idx[0] + 1, scores[0]
        s = self.item_emb[1:] @ q[0]
        top = np.argpartition(-s, n)[:n] if n < len(s) else np.arange(len(s))
        top = top[np.argsort(-s[top])]
        return top + 1, s[top]

    def _ranking_matrix(self, uid, iids):
        n = len(iids)
        hour = datetime.now().hour + 1
        base = np.stack([np.full(n, uid), iids, np.full(n, hour), np.full(n, self.default_tab)], axis=1)
        return np.concatenate([base, np.repeat(self.user_feats[uid][None, :], n, 0), self.item_feats[iids]], axis=1).astype(np.int64)

    def item_info(self, iid):
        m = self.items_meta.get(int(iid), {})
        return {"iid": int(iid), "video_id": m.get("video_id"), "author_id": m.get("author_id"), "tag": m.get("tag_first")}

    def recommend(self, uid, k=20, n_candidates=200, realtime=True):
        t0 = time.perf_counter()
        uid = int(uid)
        if not 1 <= uid <= self.meta["n_users"]:
            raise ValueError(f"uid must be in 1..{self.meta['n_users']}")

        user_state = self.store.get_user(uid) if realtime else ([], [])
        clicks = user_state[0]
        u = self.user_emb[uid].copy()
        if realtime and clicks:
            # short-term interest: pull the user vector toward what they just clicked
            u = u + self.rt_alpha * self.logic.recent_click_vector(clicks)
            u /= np.linalg.norm(u) + 1e-12
        t1 = time.perf_counter()

        seen = set(clicks)
        iids, ret_scores = self._retrieve(u, min(n_candidates + len(seen), self.n_items))
        keep = np.array([i not in seen for i in iids], dtype=bool)
        iids, ret_scores = iids[keep][:n_candidates], ret_scores[keep][:n_candidates]
        t2 = time.perf_counter()

        X = self._ranking_matrix(uid, iids)
        dense = self.logic.compute(self.store, uid, iids, user_state) if realtime else np.zeros((len(iids), 0), np.float32)
        t3 = time.perf_counter()

        scores = self.rankers["rt" if realtime else "base"].score(X, dense)
        order = np.argsort(-scores)[:k]
        t4 = time.perf_counter()

        items = []
        for r in order:
            info = self.item_info(iids[r])
            info.update(score=round(float(scores[r]), 4), retrieval_rank=int(r) + 1)
            if realtime:
                info["features"] = {"sim_recent": round(float(dense[r, 1]), 3), "tag_match": round(float(dense[r, 2]), 3)}
            items.append(info)
        ms = lambda a, b: round((b - a) * 1000, 2)
        return {
            "uid": uid, "realtime": realtime, "items": items,
            "recent_clicks": [self.item_info(i) for i in clicks[:10]],
            "timings_ms": {"user_vector": ms(t0, t1), "retrieval": ms(t1, t2), "features": ms(t2, t3),
                           "ranking": ms(t3, t4), "total": ms(t0, t4)},
            "n_candidates": int(len(iids)),
        }
