"""Shared helpers: data loading, device selection, metrics, result saving."""
import json
import os
import random

import numpy as np
import pandas as pd


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def load_processed(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    inter = pd.read_pickle(os.path.join(data_dir, "interactions.pkl"))
    rand_path = os.path.join(data_dir, "random.pkl")
    rand = pd.read_pickle(rand_path) if os.path.exists(rand_path) else None
    feats = np.load(os.path.join(data_dir, "features.npz"))
    return meta, inter, rand, feats["user_feats"], feats["item_feats"]


# ---------------- retrieval metrics ----------------
def topk_metrics(topk_items, truths, ks):
    """topk_items: [n_users, max_k] item ids ranked; truths: list of np arrays of relevant item ids.
    Full-ranking Recall@K and NDCG@K (no sampled negatives, which inflate numbers)."""
    out = {}
    for k in ks:
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        rec, ndcg = [], []
        for row, t in zip(topk_items, truths):
            hits = np.isin(row[:k], t)
            rec.append(hits.sum() / len(t))
            idcg = discounts[: min(len(t), k)].sum()
            ndcg.append((hits * discounts).sum() / idcg)
        out[f"recall@{k}"] = float(np.mean(rec)) if rec else 0.0
        out[f"ndcg@{k}"] = float(np.mean(ndcg)) if ndcg else 0.0
    return out


# ---------------- ranking metrics ----------------
def gauc(uids, labels, scores):
    """Impression-weighted per-user AUC over users that have both clicks and non-clicks.
    Closer to what matters in a feed: ranking quality *within* each user's candidates."""
    df = pd.DataFrame({"u": uids, "y": labels, "s": scores})
    df["r"] = df.groupby("u")["s"].rank(method="average")
    g = df.groupby("u").agg(n=("y", "size"), pos=("y", "sum"), rpos=("r", lambda r: r[df.loc[r.index, "y"] == 1].sum()))
    g = g[(g.pos > 0) & (g.pos < g.n)]
    if g.empty:
        return float("nan")
    neg = g.n - g.pos
    auc_u = (g.rpos - g.pos * (g.pos + 1) / 2) / (g.pos * neg)
    return float((auc_u * g.n).sum() / g.n.sum())


def ranking_metrics(uids, labels, scores):
    from sklearn.metrics import log_loss, roc_auc_score
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "gauc": gauc(uids, labels, scores),
        "logloss": float(log_loss(labels, p)),
        "n": int(len(labels)),
    }


def save_results(results_dir, name, payload):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {path}")
