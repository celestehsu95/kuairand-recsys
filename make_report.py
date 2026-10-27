"""Collect results/*.json into the Markdown tables for the README."""
import argparse
import glob
import json
import os

RETRIEVAL_ORDER = ["popularity", "two_tower_id_only", "two_tower_no_logq", "two_tower"]
RETRIEVAL_LABEL = {
    "popularity": "Popularity baseline",
    "two_tower_id_only": "Two-tower, ID embeddings only",
    "two_tower_no_logq": "Two-tower, no logQ correction",
    "two_tower": "**Two-tower (side features + logQ)**",
}
RANKING_ORDER = ["item_ctr", "dnn", "dcn"]
RANKING_LABEL = {"item_ctr": "Item CTR baseline", "dnn": "**DNN** (used downstream)", "dcn": "DCN-v2 (DNN + cross layers)"}


def fmt(x):
    return "–" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--update_readme", action="store_true", help="write the tables into README.md")
    args = ap.parse_args()
    res = {}
    for p in glob.glob(os.path.join(args.results_dir, "*.json")):
        with open(p) as f:
            res[os.path.basename(p)[:-5]] = json.load(f)

    n_cold = res.get("retrieval_two_tower", {}).get("test", {}).get("n_cold")
    cold_hdr = f"Recall@50 (cold users, n={n_cold})" if n_cold else "Recall@50 (cold users)"
    lines = ["### Retrieval (test period, full-catalog ranking)", "",
             f"| Model | Recall@50 | NDCG@50 | Recall@100 | {cold_hdr} |",
             "|---|---|---|---|---|"]
    for key in RETRIEVAL_ORDER:
        r = res.get(f"retrieval_{key}")
        if not r:
            continue
        t = r["test"]
        cold = t.get("cold", {}).get("recall@50")
        lines.append(f"| {RETRIEVAL_LABEL[key]} | {fmt(t['all'].get('recall@50'))} | {fmt(t['all'].get('ndcg@50'))} | "
                     f"{fmt(t['all'].get('recall@100'))} | {fmt(cold)} |")
    pop = res.get("retrieval_popularity", {}).get("test", {}).get("all", {}).get("recall@50")
    tt = res.get("retrieval_two_tower", {}).get("test", {}).get("all", {}).get("recall@50")
    if pop and tt:
        lines += ["", f"Two-tower vs popularity, Recall@50: **{(tt / pop - 1) * 100:+.1f}%**"]

    lines += ["", "### Ranking (test period, click prediction)", "",
              "| Model | AUC | GAUC | LogLoss | AUC on random-exposure log |", "|---|---|---|---|---|"]
    for key in RANKING_ORDER:
        r = res.get(f"ranking_{key}")
        if not r or "test_standard" not in r:
            continue
        s, rnd = r["test_standard"], r.get("test_random", {})
        lines.append(f"| {RANKING_LABEL[key]} | {fmt(s['auc'])} | {fmt(s['gauc'])} | {fmt(s['logloss'])} | {fmt(rnd.get('auc'))} |")

    rt = res.get("ranking_realtime")
    if rt and "variants" in rt:
        model = rt.get("model", "dnn").upper().replace("DCN", "DCN-v2")
        served = rt.get("served_variant")
        labels = {"base": "No real-time features", "all": "+ all real-time features",
                  "candidate": "+ candidate-level only", "user": "+ user-level only"}
        lines += ["", f"### Real-time features ({model} ranker, same data and seed)", "",
                  "| Ranker | Val GAUC | AUC | GAUC | LogLoss | AUC on random-exposure log |", "|---|---|---|---|---|---|"]
        for key, v in rt["variants"].items():
            s, rnd = v.get("test", {}), v.get("test_random", {})
            label = labels.get(key, key) + (" **(served)**" if key == served else "")
            lines.append(f"| {label} | {fmt(v['val'].get('gauc'))} | {fmt(s.get('auc'))} | {fmt(s.get('gauc'))} | "
                         f"{fmt(s.get('logloss'))} | {fmt(rnd.get('auc'))} |")
        lines += ["", "Candidate-level features vary across the items in one request; user-level features are constant "
                      "within a request. Served variant chosen by validation GAUC."]

    lat_path = os.path.join(args.results_dir, "latency.json")
    if os.path.exists(lat_path):
        with open(lat_path) as f:
            runs = json.load(f)
        lines += ["", "### Serving latency (`./demo.sh loadtest`, single machine, Docker)", "",
                  "| Concurrency | Real-time | QPS | p50 ms | p95 ms | p99 ms | Server: retrieval / features / ranking ms |",
                  "|---|---|---|---|---|---|---|"]
        for r in runs:
            s = r["server_stage_mean_ms"]
            lines.append(f"| {r['concurrency']} | {'yes' if r['realtime'] else 'no'} | {r['qps']} | {r['p50_ms']} | {r['p95_ms']} | "
                         f"{r['p99_ms']} | {s['retrieval']} / {s['features']} / {s['ranking']} |")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.results_dir, "RESULTS.md"), "w") as f:
        f.write(report + "\n")
    print(f"\nWritten to {args.results_dir}/RESULTS.md")

    if args.update_readme and os.path.exists("README.md"):
        with open("README.md") as f:
            readme = f.read()
        start, end = "<!-- RESULTS START -->", "<!-- RESULTS END -->"
        if start in readme and end in readme:
            readme = readme.split(start)[0] + start + "\n" + report + "\n" + end + readme.split(end)[1]
            with open("README.md", "w") as f:
                f.write(readme)
            print("README.md results section updated.")


if __name__ == "__main__":
    main()
