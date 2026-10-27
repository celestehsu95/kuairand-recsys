### Retrieval (test period, full-catalog ranking)

| Model | Recall@50 | NDCG@50 | Recall@100 | Recall@50 (cold users, n=209) |
|---|---|---|---|---|
| Popularity baseline | 0.1145 | 0.0433 | 0.1812 | 0.2075 |
| Two-tower, ID embeddings only | 0.1144 | 0.0434 | 0.1812 | 0.2094 |
| Two-tower, no logQ correction | 0.0524 | 0.0168 | 0.0891 | 0.0374 |
| **Two-tower (side features + logQ)** | 0.1780 | 0.0597 | 0.2865 | 0.2440 |

Two-tower vs popularity, Recall@50: **+55.4%**

### Ranking (test period, click prediction)

| Model | AUC | GAUC | LogLoss | AUC on random-exposure log |
|---|---|---|---|---|
| Item CTR baseline | 0.6742 | 0.6106 | 0.6370 | 0.5404 |
| **DNN** (used downstream) | 0.7377 | 0.6436 | 0.6011 | 0.6455 |
| DCN-v2 (DNN + cross layers) | 0.7285 | 0.6257 | 0.6035 | 0.6436 |

### Real-time features (DNN ranker, same data and seed)

| Ranker | Val GAUC | AUC | GAUC | LogLoss | AUC on random-exposure log |
|---|---|---|---|---|---|
| No real-time features | 0.6421 | 0.7370 | 0.6415 | 0.6008 | 0.6452 |
| + all real-time features | 0.6316 | 0.7454 | 0.6260 | 0.5969 | 0.6978 |
| + candidate-level only **(served)** | 0.6482 | 0.7389 | 0.6436 | 0.5899 | 0.6728 |
| + user-level only | 0.6303 | 0.7423 | 0.6263 | 0.6005 | 0.6720 |

Candidate-level features vary across the items in one request; user-level features are constant within a request. Served variant chosen by validation GAUC.

### Serving latency (`./demo.sh loadtest`, single machine, Docker)

| Concurrency | Real-time | QPS | p50 ms | p95 ms | p99 ms | Server: retrieval / features / ranking ms |
|---|---|---|---|---|---|---|
| 1 | yes | 158.1 | 5.14 | 10.22 | 19.76 | 0.22 / 1.3 / 1.81 |
| 8 | yes | 276.7 | 30.68 | 62.71 | 115.21 | 0.65 / 4.0 / 17.35 |
| 32 | yes | 214.6 | 102.06 | 475.27 | 810.8 | 0.99 / 6.55 / 39.03 |
| 8 | no | 215.9 | 27.89 | 75.68 | 222.0 | 0.88 / 1.07 / 27.6 |
