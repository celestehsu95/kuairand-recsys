"""Bundle everything the serving stack needs into artifacts/ (mounted into the containers).

    python export_artifacts.py
"""
import argparse
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--n_demo_users", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for f in ["meta.json", "features.npz", "id_maps.npz", "items_meta.csv", "emb.npz", "events.npz"]:
        shutil.copy(os.path.join(args.data_dir, f), os.path.join(args.out_dir, f))
    for f in ["ranker_base.pt", "ranker_base.json", "ranker_rt.pt", "ranker_rt.json"]:
        shutil.copy(os.path.join(args.ckpt_dir, f), os.path.join(args.out_dir, f))

    # demo users: the most active users in the test period (they make the live demo interesting)
    ev = np.load(os.path.join(args.data_dir, "events.npz"))
    test = ev["split"] == "test"
    counts = np.bincount(ev["uid"][test & (ev["is_click"] == 1)])
    demo = [int(u) for u in np.argsort(-counts)[: args.n_demo_users] if u != 0 and counts[u] > 0]
    with open(os.path.join(args.out_dir, "demo_users.json"), "w") as f:
        json.dump(demo, f)

    # serving defaults for request context the demo doesn't have (which app tab the feed is on)
    import pandas as pd
    inter = pd.read_pickle(os.path.join(args.data_dir, "interactions.pkl"))
    with open(os.path.join(args.out_dir, "serving_config.json"), "w") as f:
        json.dump({"default_tab": int(inter.tab.mode().iloc[0])}, f)

    size = sum(os.path.getsize(os.path.join(args.out_dir, f)) for f in os.listdir(args.out_dir))
    print(f"Artifacts written to {args.out_dir}/ ({size / 1e6:.0f} MB, {len(demo)} demo users)")


if __name__ == "__main__":
    main()
