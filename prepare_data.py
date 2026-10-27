"""Preprocess KuaiRand-Pure into model-ready arrays.

Outputs (in --out_dir):
  interactions.pkl  standard-feed exposures with uid/iid/hour/tab/is_click and a temporal split
  random.pkl        random-exposure log (unbiased; used for debiased evaluation of the ranker)
  features.npz      user_feats [n_users+1, F_u], item_feats [n_items+1, F_i]  (row 0 = unknown/padding)
  meta.json         vocab sizes, feature names, split dates, dataset stats

Split: temporal by date. The last --test_days dates are test, the --val_days before that are
validation, everything earlier is train. No random split -> no future leakage.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

USER_CAT_COLS = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]
ITEM_CAT_COLS = ["author_id", "video_type", "upload_type", "music_type", "tag_first", "duration_bucket"]
LOG_COLS = ["user_id", "video_id", "date", "hourmin", "time_ms", "is_click", "tab"]


def find(raw_dir, pattern):
    return sorted(glob.glob(os.path.join(raw_dir, "**", pattern), recursive=True))


def read_logs(files):
    if not files:
        return None
    frames = []
    for f in files:
        head = pd.read_csv(f, nrows=0).columns
        frames.append(pd.read_csv(f, usecols=[c for c in LOG_COLS if c in head]))
        print(f"  loaded {os.path.basename(f)}: {len(frames[-1]):,} rows")
    return pd.concat(frames, ignore_index=True)


def encode_column(series):
    """Map any column to contiguous ints starting at 1 (0 is reserved for unknown)."""
    codes, uniques = pd.factorize(series.astype(str))
    return codes + 1, len(uniques) + 1


def add_hour(df):
    if "hourmin" in df:
        df["hour"] = (df["hourmin"] // 100).clip(0, 23).astype(int) + 1
    else:
        df["hour"] = pd.to_datetime(df["time_ms"], unit="ms").dt.hour + 1
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/KuaiRand-Pure")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_days", type=int, default=3)
    ap.add_argument("--test_days", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    std_files = find(args.raw_dir, "log_standard*.csv")
    rand_files = find(args.raw_dir, "log_random*.csv")
    uf_files = find(args.raw_dir, "user_features*.csv")
    vf_files = find(args.raw_dir, "video_features_basic*.csv")
    if not std_files or not uf_files or not vf_files:
        raise SystemExit(f"Could not find KuaiRand CSVs under {args.raw_dir}. See README for the download step.")

    print("Reading logs...")
    std = read_logs(std_files)
    rand = read_logs(rand_files)
    users = pd.read_csv(uf_files[0])
    videos = pd.read_csv(vf_files[0])

    # ---- ID maps (0 reserved) ----
    all_uids = pd.Index(sorted(set(users.user_id) | set(std.user_id) | (set(rand.user_id) if rand is not None else set())))
    all_iids = pd.Index(sorted(set(videos.video_id) | set(std.video_id) | (set(rand.video_id) if rand is not None else set())))
    n_users, n_items = len(all_uids), len(all_iids)

    # ---- user side features ----
    u_cols = [c for c in USER_CAT_COLS if c in users.columns]
    user_feats = np.zeros((n_users + 1, len(u_cols)), dtype=np.int64)
    u_rows = all_uids.get_indexer(users.user_id) + 1
    user_vocabs = []
    for j, c in enumerate(u_cols):
        codes, vocab = encode_column(users[c])
        user_feats[u_rows, j] = codes
        user_vocabs.append(vocab)

    # ---- item side features ----
    videos["tag_first"] = videos["tag"].astype(str).str.split(",").str[0] if "tag" in videos else "NA"
    if "video_duration" in videos:
        videos["duration_bucket"] = pd.qcut(videos["video_duration"], 10, labels=False, duplicates="drop").fillna(-1)
    i_cols = [c for c in ITEM_CAT_COLS if c in videos.columns]
    item_feats = np.zeros((n_items + 1, len(i_cols)), dtype=np.int64)
    i_rows = all_iids.get_indexer(videos.video_id) + 1
    item_vocabs = []
    for j, c in enumerate(i_cols):
        codes, vocab = encode_column(videos[c])
        item_feats[i_rows, j] = codes
        item_vocabs.append(vocab)

    # ---- temporal split ----
    dates = sorted(std.date.unique())
    test_dates = dates[-args.test_days:]
    val_dates = dates[-(args.test_days + args.val_days):-args.test_days]

    def to_split(d):
        return np.where(d.isin(test_dates), "test", np.where(d.isin(val_dates), "val", "train"))

    has_tab = "tab" in std.columns
    tab_vocab = 1
    if has_tab:
        tab_values = pd.concat([std.tab, rand.tab]) if (rand is not None and "tab" in rand) else std.tab
        tab_index = pd.Index(sorted(tab_values.unique()))
        tab_vocab = len(tab_index) + 1

    def finalize(df):
        df = add_hour(df.copy())
        out = pd.DataFrame({
            "uid": all_uids.get_indexer(df.user_id) + 1,
            "iid": all_iids.get_indexer(df.video_id) + 1,
            "hour": df.hour.astype(np.int64),
            "tab": (tab_index.get_indexer(df.tab) + 1) if (has_tab and "tab" in df) else 0,
            "date": df.date,
            "time_ms": df.time_ms if "time_ms" in df else 0,
            "is_click": df.is_click.astype(np.int64),
            "split": to_split(df.date),
        })
        return out.sort_values(["uid", "time_ms"]).reset_index(drop=True)

    inter = finalize(std)
    inter.to_pickle(os.path.join(args.out_dir, "interactions.pkl"))
    if rand is not None:
        finalize(rand).to_pickle(os.path.join(args.out_dir, "random.pkl"))
    np.savez(os.path.join(args.out_dir, "features.npz"), user_feats=user_feats, item_feats=item_feats)
    # raw-ID maps and human-readable item metadata, used by the serving demo
    np.savez(os.path.join(args.out_dir, "id_maps.npz"), raw_uids=all_uids.to_numpy(), raw_iids=all_iids.to_numpy())
    meta_cols = [c for c in ["video_id", "author_id", "tag_first", "video_type", "video_duration"] if c in videos.columns]
    items_meta = pd.DataFrame({"iid": np.arange(1, n_items + 1), "video_id": all_iids.to_numpy()})
    items_meta = items_meta.merge(videos[meta_cols], on="video_id", how="left")
    items_meta.to_csv(os.path.join(args.out_dir, "items_meta.csv"), index=False)

    stats = {s: {"rows": int((inter.split == s).sum()),
                 "clicks": int(((inter.split == s) & (inter.is_click == 1)).sum())}
             for s in ["train", "val", "test"]}
    meta = {
        "n_users": n_users, "n_items": n_items,
        "user_feat_names": u_cols, "user_vocabs": user_vocabs,
        "item_feat_names": i_cols, "item_vocabs": item_vocabs,
        "hour_vocab": 25, "tab_vocab": tab_vocab,
        "val_dates": [int(d) for d in val_dates], "test_dates": [int(d) for d in test_dates],
        "has_random_log": rand is not None,
        "stats": stats,
        "overall_ctr": float(inter.is_click.mean()),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nUsers: {n_users:,}  Items: {n_items:,}  Standard-log exposures: {len(inter):,}  CTR: {meta['overall_ctr']:.3f}")
    print(f"Val dates: {meta['val_dates']}  Test dates: {meta['test_dates']}")
    for s, v in stats.items():
        print(f"  {s:5s}: {v['rows']:>9,} exposures, {v['clicks']:>8,} clicks")
    print(f"User features: {u_cols}\nItem features: {i_cols}")
    print(f"Saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
