"""Train the two-tower retrieval model and evaluate full-ranking Recall@K / NDCG@K.

    python train_retrieval.py                          # full model (side features + logQ)
    python train_retrieval.py --no_side_features       # ablation: ID embeddings only
    python train_retrieval.py --no_logq                # ablation: no popularity-bias correction
"""
import argparse
import copy
import os
import time

import numpy as np
import torch

from baselines import eval_retrieval, retrieval_truth, train_click_counts
from common import get_device, load_processed, save_results, set_seed, topk_metrics
from models import TwoTower


@torch.no_grad()
def recommend(model, uids, n_items, k, device, chunk=4096):
    """Exact maximum-inner-product search over the full catalog (7.5k items fits easily).
    At production scale this is where an ANN index (FAISS/ScaNN) goes."""
    model.eval()
    item_ids = torch.arange(1, n_items + 1, device=device)
    item_emb = model.item_tower(item_ids)
    out = []
    for s in range(0, len(uids), chunk):
        u = model.user_tower(torch.as_tensor(uids[s:s + chunk], device=device))
        top = (u @ item_emb.T).topk(k, dim=1).indices + 1  # back to 1-based item ids
        out.append(top.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--out_dim", type=int, default=64)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--no_logq", action="store_true")
    ap.add_argument("--no_side_features", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="name suffix for the results file")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    meta, inter, _, user_feats, item_feats = load_processed(args.data_dir)
    n_users, n_items = meta["n_users"], meta["n_items"]
    tag = args.tag or ("two_tower" + ("_id_only" if args.no_side_features else "") + ("_no_logq" if args.no_logq else ""))
    print(f"[{tag}] device={device}")

    train_pos = inter[(inter.split == "train") & (inter.is_click == 1)]
    pairs_u = torch.as_tensor(train_pos.uid.to_numpy(), device=device)
    pairs_i = torch.as_tensor(train_pos.iid.to_numpy(), device=device)
    print(f"Training pairs (clicks): {len(train_pos):,}")

    counts = train_click_counts(inter, n_items)
    log_q = None
    if not args.no_logq:
        q = (counts + 1.0) / (counts + 1.0).sum()  # +1 so unseen items get a finite correction
        log_q = torch.as_tensor(np.log(q), dtype=torch.float32, device=device)

    model = TwoTower(n_users, n_items, user_feats, item_feats, meta["user_vocabs"], meta["item_vocabs"],
                     use_side_features=not args.no_side_features, out_dim=args.out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_uids, val_truths = retrieval_truth(inter, "val")
    max_k = max(args.ks)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        perm = torch.randperm(len(pairs_u)).to(device)
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            if len(idx) < 2:
                continue
            loss = model.in_batch_loss(pairs_u[idx], pairs_i[idx], args.temperature, log_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)
        val = topk_metrics(recommend(model, val_uids, n_items, max_k, device), val_truths, args.ks)
        score = val[f"recall@{50 if 50 in args.ks else max_k}"]
        history.append({"epoch": epoch, "loss": total / max(n, 1), **val})
        print(f"epoch {epoch:2d}  loss {total / max(n, 1):.4f}  val {({k: round(v, 4) for k, v in val.items()})}  ({time.time() - t0:.0f}s)")
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    test_uids, test_truths = retrieval_truth(inter, "test")
    test = eval_retrieval(recommend(model, test_uids, n_items, max_k, device), test_uids, test_truths, inter, args.ks)
    print("TEST:", {k: round(v, 4) for k, v in test["all"].items()})

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{tag}.pt")
    save_results(args.results_dir, f"retrieval_{tag}", {"model": tag, "args": vars(args), "test": test, "history": history})


if __name__ == "__main__":
    main()
