"""Non-neural pieces: eval-set construction, feature matrices, and the two baselines.

Kept free of PyTorch so they can be tested and run on their own:
    python baselines.py --data_dir data/processed
"""
import argparse

import numpy as np

from common import load_processed, ranking_metrics, save_results, topk_metrics


# ---------------- retrieval ----------------
def retrieval_truth(inter, split):
    """Users with >=1 click in `split`, and the items they clicked."""
    pos = inter[(inter.split == split) & (inter.is_click == 1)]
    g = pos.groupby("uid")["iid"].unique()
    return g.index.to_numpy(), [np.asarray(v) for v in g.to_numpy()]


def train_click_counts(inter, n_items):
    train_pos = inter[(inter.split == "train") & (inter.is_click == 1)]
    counts = np.bincount(train_pos.iid.to_numpy(), minlength=n_items + 1).astype(np.float64)
    counts[0] = 0
    return counts


def popularity_topk(counts, k):
    order = np.argsort(-counts, kind="stable")
    return order[order != 0][:k]


def segment_users(inter, uids):
    """warm = clicked something in train; cold = no train clicks (side features carry the load)."""
    warm_set = set(inter.loc[(inter.split == "train") & (inter.is_click == 1), "uid"].unique())
    return np.array([u in warm_set for u in uids])


def eval_retrieval(topk_items, uids, truths, inter, ks):
    res = {"all": topk_metrics(topk_items, truths, ks), "n_users": int(len(uids))}
    warm = segment_users(inter, uids)
    for name, mask in [("warm", warm), ("cold", ~warm)]:
        if mask.sum() > 0:
            idx = np.where(mask)[0]
            res[name] = topk_metrics(topk_items[idx], [truths[i] for i in idx], ks)
            res[f"n_{name}"] = int(mask.sum())
    return res


# ---------------- ranking ----------------
def ranking_fields(meta):
    names = ["uid", "iid", "hour", "tab"] + [f"u_{n}" for n in meta["user_feat_names"]] + [f"i_{n}" for n in meta["item_feat_names"]]
    vocabs = [meta["n_users"] + 1, meta["n_items"] + 1, meta["hour_vocab"], meta["tab_vocab"]] + meta["user_vocabs"] + meta["item_vocabs"]
    return names, vocabs


def ranking_matrix(df, user_feats, item_feats):
    uid, iid = df.uid.to_numpy(), df.iid.to_numpy()
    base = np.stack([uid, iid, df.hour.to_numpy(), np.asarray(df.tab, dtype=np.int64)], axis=1)
    return np.concatenate([base, user_feats[uid], item_feats[iid]], axis=1).astype(np.int64)


def item_ctr_scores(train_df, eval_df, n_items, prior_strength=20.0):
    """Smoothed per-item CTR from the training period: the 'just show what gets clicked' baseline."""
    clicks = np.bincount(train_df.iid, weights=train_df.is_click, minlength=n_items + 1)
    imps = np.bincount(train_df.iid, minlength=n_items + 1)
    g = train_df.is_click.mean()
    ctr = (clicks + prior_strength * g) / (imps + prior_strength)
    return ctr[eval_df.iid.to_numpy()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100])
    args = ap.parse_args()

    meta, inter, rand, _, _ = load_processed(args.data_dir)

    # Popularity retrieval
    counts = train_click_counts(inter, meta["n_items"])
    uids, truths = retrieval_truth(inter, "test")
    top = popularity_topk(counts, max(args.ks))
    topk_items = np.tile(top, (len(uids), 1))
    pop = eval_retrieval(topk_items, uids, truths, inter, args.ks)
    print("Popularity retrieval (test):", {k: round(v, 4) for k, v in pop["all"].items()})
    save_results(args.results_dir, "retrieval_popularity", {"model": "popularity", "test": pop})

    # Item-CTR ranking baseline
    train = inter[inter.split == "train"]
    out = {"model": "item_ctr"}
    for name, df in [("test_standard", inter[inter.split == "test"]),
                     ("test_random", rand[rand.split == "test"] if rand is not None else None)]:
        if df is not None and len(df) and df.is_click.nunique() == 2:
            out[name] = ranking_metrics(df.uid.to_numpy(), df.is_click.to_numpy(), item_ctr_scores(train, df, meta["n_items"]))
            print(f"Item-CTR ranking ({name}):", {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()})
    save_results(args.results_dir, "ranking_item_ctr", out)


if __name__ == "__main__":
    main()
