"""Ranking with vs. without real-time features, plus an ablation over feature groups.

Trains the same network on identical data and seeds:
  base       categorical user/item/context fields only (what train_ranking.py does)
  all        + all 7 real-time features
  candidate  + only features that vary across candidates in one request (similarity, tag/author
             match, item recent CTR/volume). These are what can reorder a single user's feed.
  user       + only user-level features (user recent CTR / history size). Constant within a
             request, so they can raise global AUC without improving per-user ordering (GAUC).

The variant served online (exported as ranker_rt) is picked by validation GAUC, not test.

    python train_rt_ranker.py
"""
import argparse
import copy
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from baselines import ranking_fields, ranking_matrix
from common import get_device, load_processed, ranking_metrics, save_results, set_seed
from models import DCNv2
from realtime_features import FEATURE_NAMES

VARIANTS = {
    "all": FEATURE_NAMES,
    "candidate": ["sim_to_recent_clicks", "tag_match_recent", "author_match_recent", "item_recent_ctr", "item_recent_volume"],
    "user": ["user_recent_clicks", "user_recent_ctr"],
}


@torch.no_grad()
def predict(model, X, D, device, batch=16384):
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.as_tensor(X[s:s + batch], device=device)
        db = torch.as_tensor(D[s:s + batch], device=device) if model.n_dense else None
        out.append(torch.sigmoid(model(xb, db)).cpu().numpy())
    return np.concatenate(out)


def fit(name, vocabs, data, args, device, cols):
    use_rt = len(cols) > 0
    set_seed(args.seed)
    cfg = dict(emb_dim=args.emb_dim, n_cross=args.n_cross, use_cross=args.model == "dcn", n_dense=len(cols))
    model = DCNv2(vocabs, **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xtr, Dtr, ytr = data["train"]
    Xtr, Dtr, ytr = torch.as_tensor(Xtr), torch.as_tensor(np.ascontiguousarray(Dtr[:, cols])), torch.as_tensor(ytr)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            if len(idx) < 2:  # BatchNorm needs >1 row
                continue
            logits = model(Xtr[idx].to(device), Dtr[idx].to(device) if use_rt else None)
            loss = F.binary_cross_entropy_with_logits(logits, ytr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        Xv, Dv, yv, uv = data["val"]
        val = ranking_metrics(uv, yv, predict(model, Xv, Dv[:, cols], device))
        print(f"[{name}] epoch {epoch}  train_loss {total / len(perm):.4f}  val_auc {val['auc']:.4f}  ({time.time() - t0:.0f}s)")
        if val["auc"] > best:
            best, best_state, best_val, bad = val["auc"], copy.deepcopy(model.state_dict()), val, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, cfg, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--model", choices=["dcn", "dnn"], default="dnn")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb_dim", type=int, default=16)
    ap.add_argument("--n_cross", type=int, default=3)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    args = ap.parse_args()

    device = get_device()
    meta, inter, rand, user_feats, item_feats = load_processed(args.data_dir)
    rt_std = np.load(os.path.join(args.data_dir, "rt_standard.npy"))
    rt_rand = np.load(os.path.join(args.data_dir, "rt_random.npy")) if rand is not None else None
    names, vocabs = ranking_fields(meta)

    def pack(df, rt, with_uid=False):
        X = ranking_matrix(df, user_feats, item_feats)
        y = df.is_click.to_numpy().astype(np.float32)
        return (X, rt, y, df.uid.to_numpy()) if with_uid else (X, rt, y)

    data = {}
    for s in ["train", "val", "test"]:
        mask = (inter.split == s).to_numpy()
        data[s] = pack(inter[mask], rt_std[mask], with_uid=s != "train")
    if rand is not None:
        mask = (rand.split == "test").to_numpy()
        data["test_random"] = pack(rand[mask], rt_rand[mask], with_uid=True)

    out = {"model": args.model, "args": vars(args), "features": FEATURE_NAMES, "variants": {}}
    os.makedirs("checkpoints", exist_ok=True)
    runs = [("base", [])] + [(v, [FEATURE_NAMES.index(f) for f in VARIANTS[v]]) for v in args.variants]
    for name, cols in runs:
        model, cfg, val = fit(name, vocabs, data, args, device, cols)
        res = {"features": [FEATURE_NAMES[c] for c in cols], "val": val}
        for split in ["test", "test_random"]:
            if split in data:
                X, D, y, u = data[split]
                res[split] = ranking_metrics(u, y, predict(model, X, D[:, cols], device))
                print(f"[{name}] {split}: AUC {res[split]['auc']:.4f}  GAUC {res[split]['gauc']:.4f}")
        out["variants"][name] = res
        tag = "base" if name == "base" else f"rt_{name}"
        torch.save(model.state_dict(), f"checkpoints/ranker_{tag}.pt")
        with open(f"checkpoints/ranker_{tag}.json", "w") as f:
            json.dump({**cfg, "rt_feature_idx": cols, "field_names": names, "vocabs": vocabs}, f)

    # serve the real-time variant with the best *validation* GAUC (per-user ordering is what a feed needs)
    rt_names = [n for n, _ in runs if n != "base"]
    chosen = max(rt_names, key=lambda n: out["variants"][n]["val"]["gauc"])
    out["served_variant"] = chosen
    for ext in ["pt", "json"]:
        shutil.copy(f"checkpoints/ranker_rt_{chosen}.{ext}", f"checkpoints/ranker_rt.{ext}")
    print(f"\nServing variant (best val GAUC): {chosen}")
    b = out["variants"]["base"]
    for n in rt_names:
        v = out["variants"][n]
        print(f"  {n:10s} test AUC {b['test']['auc']:.4f} -> {v['test']['auc']:.4f}   GAUC {b['test']['gauc']:.4f} -> {v['test']['gauc']:.4f}")
    save_results(args.results_dir, "ranking_realtime", out)


if __name__ == "__main__":
    main()
