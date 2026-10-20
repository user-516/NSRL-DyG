"""TGN-style backbone with an optional reinforced neighbour selection gate.

The backbone supplies the ingredients a CTDG model needs: a GRU memory update, a
message that carries the neighbour memory, the centre memory, the edge feature and the
time encoding, and an MLP scorer.

`selection_mode` selects how the messages are aggregated:

    'none'  TGN's most-recent-neighbour aggregator; this is the baseline.
    'mdp'   the sequential selection of Section IV-B.  The policy walks the candidates
            in recency order, the state carries the partial aggregate h_{<i}, and the
            reward is the marginal improvement in the link-prediction score.
    'noh'   the same rollout with h_{<i} withheld from the state, which separates the
            effect of the running aggregate from the effect of the binary gate.

The gate decides which candidate messages enter the aggregation; the graph itself is
never modified.
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
        self.message_dim = message_dim
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

        # ---- reinforced selection -------------------------------------------
        # state = message_i || centre_memory || time || h_<i (two endpoints)
        self.selection_mode = 'none'
        state_dim = message_dim + 2 * emb_dim + 2 * message_dim
        self.sel = nn.Sequential(
            nn.Linear(state_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))
        for m in (self.sel[0], self.sel[2]):
            nn.init.xavier_normal_(m.weight)
        self.beta = 0.1

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

    def _messages(self, ids, timestamps):
        nb, ei, mk, dt = self._side(ids, timestamps)
        center = self.memory.emb[ids].unsqueeze(1).expand(-1, nb.size(1), -1)
        return (F.relu(self.W_msg(torch.cat([
            self.memory.emb[nb], center, self.edge_feat[ei], self.time_enc(dt)], dim=-1))),
            mk, dt)

    def _aggregate_none(self, ids, timestamps):
        msg, mk, _ = self._messages(ids, timestamps)
        # TGN's default aggregator: the most recent neighbour.  Padding sits at the front.
        return self.drop(msg[:, -1, :] * mk[:, -1:].clamp(max=1.0))

    def _score_parts(self, a, b, e, t):
        return self.score(self.drop(torch.cat([a, b, e, t], dim=-1))).squeeze(-1)

    # ------------------------------------------------------------------
    def _rollout(self, src_t, dst_t, timestamps, mem_s_old, mem_d_old, e, t_cur):
        """Sequential selection over the candidates of both endpoints."""
        K = self.n_update_neighbors
        msg_s, mk_s, dt_s = self._messages(src_t, timestamps)
        msg_d, mk_d, dt_d = self._messages(dst_t, timestamps)

        c = torch.cat([msg_s[:, -K:], msg_d[:, -K:]], dim=1)
        v = torch.cat([mk_s[:, -K:], mk_d[:, -K:]], dim=1)
        dt = torch.cat([dt_s[:, -K:], dt_d[:, -K:]], dim=1)
        order = torch.argsort(dt, dim=1)
        gather3 = lambda x: torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        c = gather3(c)
        v = torch.gather(v, 1, order)
        is_src = (order < K).float()                     # (B, 2K)
        B, N, M = c.shape

        h_s = torch.zeros(B, M, device=self.device)
        h_d = torch.zeros(B, M, device=self.device)
        cnt_s = torch.zeros(B, 1, device=self.device)
        cnt_d = torch.zeros(B, 1, device=self.device)
        states = [(h_s, h_d)]
        log_pi, ent = [], []
        for i in range(N):
            agg_s = h_s / cnt_s.clamp(min=1.0)
            agg_d = h_d / cnt_d.clamp(min=1.0)
            h_ctx = torch.zeros_like(torch.cat([agg_s, agg_d], dim=-1)) \
                if self.selection_mode == 'noh' else torch.cat([agg_s, agg_d], dim=-1)
            state = torch.cat([c[:, i], mem_s_old, t_cur, h_ctx], dim=-1).detach()
            p = torch.sigmoid(self.sel(state).squeeze(-1)).clamp(1e-6, 1 - 1e-6)
            with torch.no_grad():
                a = (torch.bernoulli(p.detach()) if self.training
                     else (p.detach() >= 0.5).float()) * v[:, i]
                w = (a * is_src[:, i]).unsqueeze(-1)
                h_s = h_s + w * c[:, i]
                cnt_s = cnt_s + w[:, :1]
                w = (a * (1 - is_src[:, i])).unsqueeze(-1)
                h_d = h_d + w * c[:, i]
                cnt_d = cnt_d + w[:, :1]
                states.append((h_s, h_d))
            log_pi.append((torch.log(p) * a + torch.log(1 - p) * (1 - a)) * v[:, i])
            ent.append(-(p * torch.log(p) + (1 - p) * torch.log(1 - p)) * v[:, i])

        with torch.no_grad():
            # score in [0,1]; a rising score means a better aggregate, so the marginal
            # improvement is L[:, i+1] - L[:, i].
            L = torch.stack([self._score_parts(hs / torch.clamp(cnt_s, min=1.0),
                                               hd / torch.clamp(cnt_d, min=1.0), e, t_cur)
                             for hs, hd in states], 1)
            returns = L[:, 1:] - L[:, :-1]

        agg_s = h_s / cnt_s.clamp(min=1.0)
        agg_d = h_d / cnt_d.clamp(min=1.0)
        return agg_s, agg_d, torch.stack(log_pi, 1), torch.stack(ent, 1), returns

    # ------------------------------------------------------------------
    def forward(self, src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps):
        src_t = torch.as_tensor(src_idxs, dtype=torch.long, device=self.device)
        dst_t = torch.as_tensor(dst_idxs, dtype=torch.long, device=self.device)
        neg_t = torch.as_tensor(neg_idxs, dtype=torch.long, device=self.device)
        eidx = torch.as_tensor(edge_idxs, dtype=torch.long, device=self.device)
        B = src_t.size(0)

        mem_s_old = self.memory.emb[src_t].clone()
        mem_d_old = self.memory.emb[dst_t].clone()
        mem_n_old = self.memory.emb[neg_t].clone()
        e = self.edge_feat[eidx]
        t_cur = self.time_enc(torch.zeros(B, device=self.device))

        policy_loss = torch.zeros((), device=self.device)
        reward = torch.zeros((), device=self.device)
        if self.selection_mode == 'none':
            agg_s = self._aggregate_none(src_t, timestamps)
            agg_d = self._aggregate_none(dst_t, timestamps)
        else:
            agg_s, agg_d, log_pi, ent, returns = self._rollout(
                src_t, dst_t, timestamps, mem_s_old, mem_d_old, e, t_cur)
            if self.training:
                adv = returns
                sd = adv.std()
                adv = adv / sd.clamp(min=1e-6) if sd > 1e-6 else adv * 0.0
                adv = adv.clamp(-3.0, 3.0)
                policy_loss = (-(log_pi * adv).sum() - self.beta * ent.sum()) / B
            reward = returns.mean().detach()
        agg_n = self._aggregate_none(neg_t, timestamps)

        rep_s = self.gru(agg_s, mem_s_old)
        rep_d = self.gru(agg_d, mem_d_old)
        rep_n = self.gru(agg_n, mem_n_old)

        pos = torch.sigmoid(self._score_parts(rep_s, rep_d, e, t_cur))
        neg = torch.sigmoid(self._score_parts(rep_s, rep_n, e, t_cur))

        with torch.no_grad():
            self.memory.emb[src_t] = rep_s.detach()
            self.memory.emb[dst_t] = rep_d.detach()

        return pos, neg, policy_loss, reward

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
