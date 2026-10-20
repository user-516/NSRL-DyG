"""A TGN-style backbone, used as a known-good reference for the data pipeline.

Why this file exists
--------------------
The legacy model tops out around 0.84 AUC on UCI, while published CTDG models report
0.92-0.95 on the same dataset.  Before attributing that gap to the proposed modules,
it helps to have a backbone with the standard ingredients (a GRU memory update, a
neighbour message that carries the edge feature and the time encoding, and an MLP
scorer).  If this backbone reaches the published range, the pipeline is sound and the
gap lies in the model; if it does not, the pipeline itself is the problem.

It exposes the same interface as models/NSRLDyG.py so trainAUC.py can drive either.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import Memory


class TimeEncode(nn.Module):
    """cos(t * w) with w = 10^{-j/d}, built in float64 to avoid overflow."""

    def __init__(self, d):
        super().__init__()
        w = 1.0 / (10.0 ** np.arange(d, dtype=np.float64))
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32).view(1, d),
                              requires_grad=False)

    def forward(self, t):
        return torch.cos(t.unsqueeze(-1) * self.w)


class TGNBackbone(nn.Module):
    def __init__(self, n_nodes, n_neighbors, n_update_neighbors,
                 edge_dim, emb_dim, message_dim,
                 neighbor_finder, edge_feat, tot_time, device='cpu'):
        super().__init__()
        self.n_nodes = n_nodes
        self.emb_dim = emb_dim
        self.n_neighbors = n_neighbors
        self.n_update_neighbors = n_update_neighbors
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.tot_time = tot_time

        self.edge_feat = torch.tensor(edge_feat, dtype=torch.float32).to(device)
        self.memory = Memory(n_nodes, emb_dim, device)
        self.time_enc = TimeEncode(emb_dim)
        self.drop = nn.Dropout(0.2)

        msg_in = 2 * emb_dim + edge_dim + emb_dim      # mem_i || mem_center || edge || time
        self.W_msg = nn.Linear(msg_in, message_dim)
        self.gru = nn.GRUCell(message_dim, emb_dim)
        self.score = nn.Sequential(
            nn.Linear(2 * emb_dim + edge_dim + emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, 1))
        for m in (self.W_msg, self.score[0], self.score[2]):
            nn.init.xavier_normal_(m.weight)

    # ------------------------------------------------------------------
    def _side(self, ids, timestamps):
        N = self.n_neighbors
        nb, ei, et, mk = self.neighbor_finder.get_temporal_neighbor(
            ids, timestamps, N, return_mask=True)
        nb_t = torch.as_tensor(nb, dtype=torch.long, device=self.device)
        ei_t = torch.as_tensor(np.clip(ei, 0, len(self.edge_feat) - 1),
                               dtype=torch.long, device=self.device)
        mk_t = torch.as_tensor(mk, dtype=torch.float32, device=self.device)
        cur = np.repeat(np.expand_dims(timestamps, 1), N, 1)
        dt = torch.as_tensor((cur - et) / self.tot_time, dtype=torch.float32, device=self.device)
        return nb_t, ei_t, mk_t, dt

    def _aggregate(self, ids, timestamps):
        """Mean of the per-neighbour messages, Eq (3) of TGN."""
        nb, ei, mk, dt = self._side(ids, timestamps)
        center = self.memory.emb[ids].unsqueeze(1).expand(-1, nb.size(1), -1)
        msg = F.relu(self.W_msg(torch.cat([
            self.memory.emb[nb], center, self.edge_feat[ei], self.time_enc(dt)], dim=-1)))
        # TGN's default aggregator keeps the most recent message; the neighbour array
        # is time-ordered with padding at the front, so the last column is the newest.
        agg = msg[:, -1, :] * mk[:, -1:].clamp(max=1.0)
        return self.drop(agg)

    def forward(self, src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps):
        src_t = torch.as_tensor(src_idxs, dtype=torch.long, device=self.device)
        dst_t = torch.as_tensor(dst_idxs, dtype=torch.long, device=self.device)
        neg_t = torch.as_tensor(neg_idxs, dtype=torch.long, device=self.device)
        eidx = torch.as_tensor(edge_idxs, dtype=torch.long, device=self.device)

        mem_s_old = self.memory.emb[src_t].clone()
        mem_d_old = self.memory.emb[dst_t].clone()
        mem_n_old = self.memory.emb[neg_t].clone()

        agg_s = self._aggregate(src_t, timestamps)
        agg_d = self._aggregate(dst_t, timestamps)
        rep_s = self.gru(agg_s, mem_s_old)
        rep_d = self.gru(agg_d, mem_d_old)

        e = self.edge_feat[eidx]
        t = self.time_enc(torch.zeros(eidx.size(0), device=self.device))

        def _score(a, b):
            return torch.sigmoid(self.score(self.drop(torch.cat([a, b, e, t], dim=-1))).squeeze(-1))

        pos = _score(rep_s, rep_d)
        neg = _score(rep_s, self.gru(self._aggregate(neg_t, timestamps), mem_n_old))

        with torch.no_grad():
            self.memory.emb[src_t] = rep_s.detach()
            self.memory.emb[dst_t] = rep_d.detach()

        zero = torch.zeros((), device=self.device)
        return pos, neg, zero, zero

    # ------------------------------------------------------------------
    def compute_score(self, src_idxs, dst_idxs, neg_idxs, timestamps=None):
        pos = torch.sigmoid((self.memory.emb[src_idxs] * self.memory.emb[dst_idxs]).sum(-1))
        neg = torch.sigmoid((self.memory.emb[src_idxs] * self.memory.emb[neg_idxs]).sum(-1))
        return pos, neg

    def reset_graph(self):
        self.memory.reset()

    def set_neighbor_finder(self, finder):
        self.neighbor_finder = finder

    def detach_memory(self):
        self.memory.detach_memory()

    def back_up_memory(self):
        return self.memory.emb.clone()

    def restore_memory(self, back_up):
        self.memory.emb = nn.Parameter(back_up.detach().clone(), requires_grad=False)
