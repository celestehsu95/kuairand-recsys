"""Point-in-time backfill of real-time features for every logged exposure.

Standard and random-exposure logs are merged into one stream (that's what a user actually
experienced) and replayed in event-time order. For each exposure we compute features from the
state *before* it, then record it. A feature therefore never sees its own label or anything
from the future: the offline equivalent of what the online feature store would have returned.

Also exports the two-tower user/item embeddings (needed by the similarity feature and by serving).

    python build_rt_features.py
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from common import load_processed
from realtime_features import N_FEATURES, FeatureLogic, InMemoryStore


def export_embeddings(meta, user_feats, item_feats, ckpt, out_path):
    import torch
    from models import TwoTower

    state = torch.load(ckpt, map_location="cpu")
    model = TwoTower(meta["n_users"], meta["n_items"], user_feats, item_feats, meta["user_vocabs"], meta["item_vocabs"],
                     out_dim=state["user_tower.net.6.weight"].shape[0] if "user_tower.net.6.weight" in state else 64)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        user_emb = model.user_tower(torch.arange(0, meta["n_users"] + 1)).numpy()
        item_emb = model.item_tower(torch.arange(0, meta["n_items"] + 1)).numpy()
    user_emb[0] = 0
    item_emb[0] = 0
    np.savez(out_path, user_emb=user_emb.astype(np.float32), item_emb=item_emb.astype(np.float32))
    print(f"Embeddings saved to {out_path}: users {user_emb.shape}, items {item_emb.shape}")


def merged_stream(inter, rand):
    parts = [inter.assign(src=0, row=np.arange(len(inter)))]
    if rand is not None:
        parts.append(rand.assign(src=1, row=np.arange(len(rand))))
    ev = pd.concat(parts, ignore_index=True)
    return ev.sort_values("time_ms", kind="mergesort").reset_index(drop=True)


def item_codes(meta, item_feats, name):
    names = meta["item_feat_names"]
    return item_feats[:, names.index(name)] if name in names else np.zeros(len(item_feats), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--two_tower_ckpt", default="checkpoints/two_tower.pt")
    args = ap.parse_args()

    meta, inter, rand, user_feats, item_feats = load_processed(args.data_dir)
    emb_path = os.path.join(args.data_dir, "emb.npz")
    if os.path.exists(args.two_tower_ckpt):
        export_embeddings(meta, user_feats, item_feats, args.two_tower_ckpt, emb_path)
    elif not os.path.exists(emb_path):
        raise SystemExit(f"Need {args.two_tower_ckpt} (run train_retrieval.py first) or a precomputed {emb_path}")
    item_emb = np.load(emb_path)["item_emb"]

    logic = FeatureLogic(item_emb, item_codes(meta, item_feats, "tag_first"),
                         item_codes(meta, item_feats, "author_id"), meta["overall_ctr"])
    store = InMemoryStore()
    ev = merged_stream(inter, rand)
    uids, iids, clicks = ev.uid.to_numpy(), ev.iid.to_numpy(), ev.is_click.to_numpy()
    src, row = ev.src.to_numpy(), ev.row.to_numpy()

    out = [np.zeros((len(inter), N_FEATURES), np.float32),
           np.zeros((len(rand) if rand is not None else 0, N_FEATURES), np.float32)]
    t0 = time.time()
    for n in range(len(ev)):
        u, i = int(uids[n]), int(iids[n])
        out[src[n]][row[n]] = logic.compute(store, u, [i])[0]   # features BEFORE seeing this event
        logic.update(store, u, i, clicks[n])                    # then record it
        if (n + 1) % 200_000 == 0:
            print(f"  {n + 1:,}/{len(ev):,} events  ({time.time() - t0:.0f}s)")

    np.save(os.path.join(args.data_dir, "rt_standard.npy"), out[0])
    if rand is not None:
        np.save(os.path.join(args.data_dir, "rt_random.npy"), out[1])
    # the merged stream is also what the serving demo bootstraps and replays from
    np.savez(os.path.join(args.data_dir, "events.npz"), uid=uids, iid=iids, is_click=clicks,
             time_ms=ev.time_ms.to_numpy(), split=ev.split.to_numpy().astype(str))
    print(f"Backfilled {len(ev):,} events in {time.time() - t0:.0f}s -> {args.data_dir}/rt_*.npy")


if __name__ == "__main__":
    main()
