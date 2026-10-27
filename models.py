"""Two-tower retrieval model and DCN-v2 ranking model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, hidden, out_dim=None, dropout=0.1):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    if out_dim is not None:
        layers.append(nn.Linear(d, out_dim))
        d = out_dim
    return nn.Sequential(*layers), d


# ============================ Retrieval ============================
class Tower(nn.Module):
    """ID embedding + categorical side-feature embeddings -> MLP -> L2-normalized vector.

    The side-feature table lives on the module as a buffer, so the tower takes only IDs.
    Side features are what let the model say something about users/items with little history.
    """

    def __init__(self, n_ids, feat_table, feat_vocabs, id_dim=64, feat_dim=16,
                 hidden=(256, 128), out_dim=64, dropout=0.1):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids + 1, id_dim)
        self.register_buffer("feat_table", torch.as_tensor(feat_table, dtype=torch.long))
        self.feat_embs = nn.ModuleList([nn.Embedding(v, feat_dim) for v in feat_vocabs])
        in_dim = id_dim + feat_dim * len(feat_vocabs)
        self.net, _ = mlp(in_dim, hidden, out_dim, dropout)

    def forward(self, ids):
        parts = [self.id_emb(ids)]
        if len(self.feat_embs):
            f = self.feat_table[ids]
            parts += [emb(f[:, j]) for j, emb in enumerate(self.feat_embs)]
        return F.normalize(self.net(torch.cat(parts, dim=-1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, user_feats, item_feats, user_vocabs, item_vocabs,
                 use_side_features=True, **kw):
        super().__init__()
        if not use_side_features:
            user_feats, item_feats = user_feats[:, :0], item_feats[:, :0]
            user_vocabs, item_vocabs = [], []
        self.user_tower = Tower(n_users, user_feats, user_vocabs, **kw)
        self.item_tower = Tower(n_items, item_feats, item_vocabs, **kw)

    def in_batch_loss(self, uids, iids, temperature=0.05, log_q=None):
        """Sampled softmax with in-batch negatives.

        Every other positive item in the batch acts as a negative. Popular items show up as
        negatives more often, so the model learns to under-score them; logQ correction subtracts
        log(sampling prob) from each logit to undo that bias (Yi et al., RecSys 2019).
        """
        u = self.user_tower(uids)
        v = self.item_tower(iids)
        logits = u @ v.T / temperature
        if log_q is not None:
            logits = logits - log_q[iids].unsqueeze(0)
        # the same item can appear twice in a batch; don't treat a copy of the positive as a negative
        same_item = iids.unsqueeze(0) == iids.unsqueeze(1)
        eye = torch.eye(len(iids), dtype=torch.bool, device=iids.device)
        logits = logits.masked_fill(same_item & ~eye, float("-inf"))
        labels = torch.arange(len(iids), device=iids.device)
        return F.cross_entropy(logits, labels)


# ============================ Ranking ============================
class CrossLayer(nn.Module):
    """DCN-v2 cross layer: x_{l+1} = x0 * (W x_l + b) + x_l  (explicit bounded-degree feature crosses)."""

    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x0, x):
        return x0 * self.lin(x) + x


class DCNv2(nn.Module):
    """Parallel DCN-v2 (Wang et al., WWW 2021). With use_cross=False it reduces to a plain DNN,
    which gives a clean ablation of what the explicit crosses add."""

    def __init__(self, field_vocabs, emb_dim=16, n_cross=3, hidden=(256, 128), dropout=0.1, use_cross=True, n_dense=0):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, emb_dim) for v in field_vocabs])
        self.n_dense = n_dense
        self.dense_norm = nn.BatchNorm1d(n_dense) if n_dense else None
        d0 = emb_dim * len(field_vocabs) + n_dense
        self.use_cross = use_cross
        self.cross = nn.ModuleList([CrossLayer(d0) for _ in range(n_cross)]) if use_cross else nn.ModuleList()
        self.deep, d_deep = mlp(d0, hidden, None, dropout)
        self.out = nn.Linear(d_deep + (d0 if use_cross else 0), 1)

    def forward(self, x, dense=None):
        parts = [emb(x[:, j]) for j, emb in enumerate(self.embs)]
        if self.n_dense:
            parts.append(self.dense_norm(dense))  # real-time features (continuous)
        x0 = torch.cat(parts, dim=-1)
        h = self.deep(x0)
        if self.use_cross:
            xc = x0
            for layer in self.cross:
                xc = layer(x0, xc)
            h = torch.cat([xc, h], dim=-1)
        return self.out(h).squeeze(-1)
