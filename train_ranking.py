"""Train the click-through ranking model (DCN-v2) and evaluate AUC / GAUC / LogLoss.

Evaluated twice on the test period:
  test_standard  exposures chosen by Kuaishou's production recommender (biased toward what it liked)
  test_random    randomly exposed videos (unbiased) -- how the model ranks items the old system wouldn't show

    python train_ranking.py --model dcn
    python train_ranking.py --model dnn     # ablation: same network without cross layers
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from baselines import ranking_fields, ranking_matrix
from common import get_device, load_processed, ranking_metrics, save_results, set_seed
from models import DCNv2


@torch.no_grad()
def predict(model, X, device, batch=16384):
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.as_tensor(X[s:s + batch], device=device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--model", choices=["dcn", "dnn"], default="dcn")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb_dim", type=int, default=16)
    ap.add_argument("--n_cross", type=int, default=3)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    meta, inter, rand, user_feats, item_feats = load_processed(args.data_dir)
    names, vocabs = ranking_fields(meta)
    print(f"[{args.model}] device={device}  fields={len(names)}")

    splits = {s: inter[inter.split == s] for s in ["train", "val", "test"]}
    X = {s: ranking_matrix(df, user_feats, item_feats) for s, df in splits.items()}
    y = {s: df.is_click.to_numpy().astype(np.float32) for s, df in splits.items()}
    print({s: f"{len(v):,} rows" for s, v in X.items()})

    model = DCNv2(vocabs, emb_dim=args.emb_dim, n_cross=args.n_cross, use_cross=args.model == "dcn").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    Xtr = torch.as_tensor(X["train"])
    ytr = torch.as_tensor(y["train"])
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val = ranking_metrics(splits["val"].uid.to_numpy(), y["val"], predict(model, X["val"], device))
        history.append({"epoch": epoch, "train_loss": total / len(perm), **val})
        print(f"epoch {epoch}  train_loss {total / len(perm):.4f}  val_auc {val['auc']:.4f}  val_gauc {val['gauc']:.4f}  ({time.time() - t0:.0f}s)")
        if val["auc"] > best:
            best, best_state, bad = val["auc"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping (ranking models usually overfit after 1-2 epochs on ID embeddings).")
                break

    model.load_state_dict(best_state)
    out = {"model": args.model, "args": vars(args), "history": history}
    eval_sets = [("test_standard", splits["test"])]
    if rand is not None:
        eval_sets.append(("test_random", rand[rand.split == "test"]))
    for name, df in eval_sets:
        if len(df) and df.is_click.nunique() == 2:
            scores = predict(model, ranking_matrix(df, user_feats, item_feats), device)
            out[name] = ranking_metrics(df.uid.to_numpy(), df.is_click.to_numpy(), scores)
            print(f"TEST {name}:", {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()})

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/ranking_{args.model}.pt")
    save_results(args.results_dir, f"ranking_{args.model}", out)


if __name__ == "__main__":
    main()
