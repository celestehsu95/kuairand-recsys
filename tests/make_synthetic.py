"""Generate a small fake dataset with the KuaiRand-Pure file layout and schema.

Used by smoke_test.sh to check the whole pipeline runs in about a minute, before the real
dataset is downloaded. Clicks depend on hidden user/item preferences, so models should beat
the baselines here too.
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/synthetic/KuaiRand-Pure/data")
    ap.add_argument("--n_users", type=int, default=1500)
    ap.add_argument("--n_items", type=int, default=600)
    ap.add_argument("--rows", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    n_topics = 8
    user_ids = np.arange(args.n_users) + 1000
    item_ids = np.arange(args.n_items) + 5000
    user_topic = rng.integers(0, n_topics, args.n_users)
    item_topic = rng.integers(0, n_topics, args.n_items)
    item_pop = rng.pareto(1.5, args.n_items) + 1

    users = pd.DataFrame({
        "user_id": user_ids,
        "user_active_degree": rng.choice(["high_active", "full_active", "middle_active", "UNKNOWN"], args.n_users),
        "is_lowactive_period": rng.integers(0, 2, args.n_users),
        "is_live_streamer": rng.integers(0, 2, args.n_users),
        "is_video_author": rng.integers(0, 2, args.n_users),
        "follow_user_num_range": rng.choice(["0", "(0,10]", "(10,50]", "(50,100]"], args.n_users),
        "fans_user_num_range": rng.choice(["0", "[1,10)", "[10,100)"], args.n_users),
        "friend_user_num_range": rng.choice(["0", "[1,5)", "[5,30)"], args.n_users),
        "register_days_range": rng.choice(["31-60", "61-90", "91-180", "181-365", "366-730"], args.n_users),
    })
    for i in range(18):
        # onehot_feat0 leaks the hidden topic, so side features genuinely help
        users[f"onehot_feat{i}"] = user_topic if i == 0 else rng.integers(0, 5, args.n_users)
    users.to_csv(os.path.join(args.out_dir, "user_features_pure.csv"), index=False)

    videos = pd.DataFrame({
        "video_id": item_ids,
        "author_id": rng.integers(0, 200, args.n_items),
        "video_type": rng.choice(["NORMAL", "AD"], args.n_items, p=[0.95, 0.05]),
        "upload_dt": "2022-04-01",
        "upload_type": rng.choice(["ShortImport", "Web", "Kmovie"], args.n_items),
        "visible_status": "public",
        "video_duration": rng.integers(5_000, 120_000, args.n_items).astype(float),
        "music_id": rng.integers(0, 300, args.n_items),
        "music_type": rng.choice([0, 4, 9, 11], args.n_items),
        "tag": [f"{t},{rng.integers(0, 60)}" for t in item_topic],  # first tag = topic
    })
    videos.loc[rng.random(args.n_items) < 0.02, "video_duration"] = np.nan
    videos.to_csv(os.path.join(args.out_dir, "video_features_basic_pure.csv"), index=False)

    dates = pd.date_range("2022-04-08", "2022-05-08").strftime("%Y%m%d").astype(int).to_numpy()

    def make_log(n, random_exposure):
        u = rng.integers(0, args.n_users, n)
        if random_exposure:
            it = rng.integers(0, args.n_items, n)
        else:  # production feed: popularity- and topic-biased exposure
            p = item_pop / item_pop.sum()
            it = rng.choice(args.n_items, n, p=p)
            match = rng.random(n) < 0.4
            same_topic = [np.flatnonzero(item_topic == t) for t in range(n_topics)]
            it[match] = [rng.choice(same_topic[t]) if len(same_topic[t]) else 0 for t in user_topic[u[match]]]
        logit = -2.0 + 2.2 * (user_topic[u] == item_topic[it]) + 0.3 * np.log(item_pop[it])
        click = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        d = rng.choice(dates if not random_exposure else dates[dates >= 20220422], n)
        hm = rng.integers(0, 24, n) * 100 + rng.integers(0, 60, n)
        base_ms = pd.to_datetime(d.astype(str)).astype("int64") // 10**6
        return pd.DataFrame({
            "user_id": user_ids[u], "video_id": item_ids[it], "date": d, "hourmin": hm,
            "time_ms": base_ms + (hm // 100) * 3_600_000 + (hm % 100) * 60_000,
            "is_click": click, "is_like": 0, "is_follow": 0, "is_comment": 0, "is_forward": 0,
            "is_hate": 0, "long_view": click, "play_time_ms": click * 10_000, "duration_ms": 30_000,
            "profile_stay_time": 0, "comment_stay_time": 0, "is_profile_enter": 0,
            "is_rand": int(random_exposure), "tab": rng.integers(0, 4, n),
        })

    std = make_log(args.rows, False)
    std[std.date <= 20220421].to_csv(os.path.join(args.out_dir, "log_standard_4_08_to_4_21_pure.csv"), index=False)
    std[std.date > 20220421].to_csv(os.path.join(args.out_dir, "log_standard_4_22_to_5_08_pure.csv"), index=False)
    make_log(args.rows // 5, True).to_csv(os.path.join(args.out_dir, "log_random_4_22_to_5_08_pure.csv"), index=False)
    print(f"Synthetic KuaiRand-Pure written to {args.out_dir}")


if __name__ == "__main__":
    main()
