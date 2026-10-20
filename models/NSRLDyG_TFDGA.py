import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory import Memory

class TimeFrequencyEncode(nn.Module):

    def __init__(self, d, alpha=10.0, beta=10.0, mu=1.15, delta=10.0):
        super().__init__()
        j = torch.arange(d, dtype=torch.float32)
        self.register_buffer('omega', alpha ** (-j / beta))
        self.register_buffer('kappa', mu ** (j / delta))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.d = d

    def forward(self, dt, freq, f_max):
        return (torch.cos(dt.unsqueeze(-1) * self.omega) + self.gamma * torch.sin(math.pi / 2.0 * (freq.unsqueeze(-1) / (f_max + 1e-08)) * self.kappa)) / math.sqrt(self.d)

class TemporalNeighborAggregation(nn.Module):

    def __init__(self, d, d_half):
        super().__init__()
        self.W_g = nn.Linear(2 * d, d)
        self.a_alpha = nn.Parameter(torch.randn(d) * 0.1)
        self.W_m = nn.Linear(d, d_half, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.phi = TimeFrequencyEncode(d)
        for m in (self.W_g, self.W_m):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x_center, x_i, dt, freq, f_max, mask):
        phi = self.phi(dt, freq, f_max)
        z = phi * x_i
        pair = torch.cat([x_center.unsqueeze(1).expand(-1, z.size(1), -1), z], dim=-1)
        score = self.leaky(self.W_g(pair)) @ self.a_alpha
        score = score.masked_fill(mask <= 0, -1000000000.0)
        alpha = torch.softmax(score, dim=1) * mask
        seq = alpha.unsqueeze(-1) * self.W_m(z)
        return (seq, alpha)

class AdaptiveTemporalPartition(nn.Module):

    def __init__(self, gamma0=0.5, lam=0.1, l_min=2, l_max=20, temp=1.0):
        super().__init__()
        self.W_delta = nn.Linear(2, 1)
        nn.init.xavier_normal_(self.W_delta.weight)
        self.gamma0, self.lam, self.l_min, self.l_max, self.temp = (gamma0, lam, l_min, l_max, temp)

    def forward(self, seq, dt, mask):
        B, N, _ = seq.shape
        prev = torch.cat([seq.new_zeros(B, 1, seq.size(-1)), seq[:, :-1]], dim=1)
        drift = 1.0 - F.cosine_similarity(seq, prev, dim=-1)
        delta = torch.sigmoid(self.W_delta(torch.stack([drift, dt], -1)).squeeze(-1)) * mask
        S = torch.cumsum(delta, dim=1)
        l = torch.arange(1, N + 1, device=seq.device).float().view(1, N)
        Gamma = self.gamma0 * torch.exp(self.lam * torch.clamp(self.l_min - l, min=0.0)) * (1.0 - l / self.l_max)
        p = torch.sigmoid((S - Gamma) / self.temp) * mask
        cum = torch.cumsum(torch.log1p(-p.clamp(max=1 - 1e-06)), dim=1)
        cum = torch.cat([torch.zeros(B, 1, device=seq.device), cum], dim=1)
        log_m = cum[:, 1:].unsqueeze(1) - cum[:, 1:].unsqueeze(2)
        iu = torch.triu(torch.ones(N, N, device=seq.device, dtype=torch.bool), 1)
        M_l = torch.exp(log_m.clamp(min=-30.0, max=0.0)) * iu.unsqueeze(0)
        return (M_l, (1.0 - M_l) * iu.unsqueeze(0))

class DualGranularityFusion(nn.Module):

    def __init__(self, d_half, heads=4, max_len=256):
        super().__init__()
        assert d_half % heads == 0
        self.d, self.K, self.dh = (d_half, heads, d_half // heads)
        self.W_P = nn.Linear(d_half, d_half)
        self.pos = nn.Parameter(torch.zeros(max_len, d_half))
        nn.init.normal_(self.pos, std=0.02)
        self.W_Q, self.W_K, self.W_V = (nn.Linear(d_half, d_half) for _ in range(3))
        self.W_s = nn.Linear(d_half, d_half)
        self.W_d = nn.Linear(2 * d_half, d_half)
        self.W_Qc, self.W_Kc, self.W_Vc = (nn.Linear(d_half, d_half) for _ in range(3))
        self.norm = nn.LayerNorm(d_half)
        for m in (self.W_P, self.W_Q, self.W_K, self.W_V, self.W_s, self.W_d, self.W_Qc, self.W_Kc, self.W_Vc):
            nn.init.xavier_normal_(m.weight)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.K, self.dh).transpose(1, 2)

    def _merge(self, x):
        B, K, N, dh = x.shape
        return x.transpose(1, 2).reshape(B, N, K * dh)

    def forward(self, seq, M_l, M_g, mask):
        B, N, _ = seq.shape
        H = F.gelu(self.W_P(seq)) + self.pos[:N].unsqueeze(0)
        Q, K, V = (self._split(self.W_Q(H)), self._split(self.W_K(H)), self._split(self.W_V(H)))
        A = Q @ K.transpose(-1, -2) / math.sqrt(self.dh) + torch.log(M_g + 1e-09).unsqueeze(1)
        H_s = F.relu(self.W_s(self._merge(torch.softmax(A, -1) @ V)))
        X_aug = self.W_d(torch.cat([H_s, H], dim=-1))
        Qc = self._split(self.W_Qc(H_s))
        Kc = self._split(self.W_Kc(X_aug))
        Vc = self._split(self.W_Vc(X_aug))
        Ac = Qc @ Kc.transpose(-1, -2) / math.sqrt(self.d) + torch.log(M_l + 1e-09).unsqueeze(1)
        out = self._merge(torch.softmax(Ac, -1) @ Vc)
        return self.norm(out + seq) * mask.unsqueeze(-1)

class TFDDGANet(nn.Module):

    def __init__(self, n_nodes, n_neighbors, n_update_neighbors, edge_dim, emb_dim, message_dim, neighbor_finder, edge_feat, tot_time, device='cpu'):
        super().__init__()
        self.n_nodes = n_nodes
        self.emb_dim, self.message_dim = (emb_dim, message_dim)
        self.d_half = message_dim // 2
        self.n_neighbors = n_neighbors
        self.n_update_neighbors = n_update_neighbors
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.tot_time = tot_time
        self.gamma = 1.0
        self.edge_feat = torch.tensor(edge_feat, dtype=torch.float32).to(device)
        self.memory = Memory(n_nodes, emb_dim, device)
        self.aggregator = TemporalNeighborAggregation(emb_dim, self.d_half)
        self.partition = AdaptiveTemporalPartition()
        self.fusion = DualGranularityFusion(self.d_half)
        self.W_e = nn.Linear(edge_dim, message_dim)
        self.msg_dim = self.d_half * 2 + message_dim
        self.W_uc = nn.Linear(emb_dim + self.msg_dim, emb_dim)
        self.gru = nn.GRUCell(self.msg_dim, emb_dim)
        self.score_mlp = nn.Sequential(nn.Linear(2 * emb_dim + message_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))
        for m in (self.W_e, self.W_uc, self.score_mlp[0], self.score_mlp[2]):
            nn.init.xavier_normal_(m.weight)
        self.selection_mode = 'none'
        self.beta = 0.1
        state_dim = self.d_half + 2 * emb_dim + 2 * self.d_half
        self.sel = nn.Sequential(nn.Linear(state_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))
        for m in (self.sel[0], self.sel[2]):
            nn.init.xavier_normal_(m.weight)
        self.register_buffer('sel_omega', 1.0 / 10.0 ** (torch.arange(emb_dim).float() / emb_dim))
        self.beta = 0.1

    def _side(self, ids, timestamps):
        N = self.n_neighbors
        nb, _, et, mk = self.neighbor_finder.get_temporal_neighbor(ids, timestamps, N, return_mask=True)
        nb_t = torch.as_tensor(nb, dtype=torch.long, device=self.device)
        mk_t = torch.as_tensor(mk, dtype=torch.float32, device=self.device)
        cur = np.repeat(np.expand_dims(timestamps, 1), N, 1)
        dt = torch.as_tensor((cur - et) / self.tot_time, dtype=torch.float32, device=self.device)
        return (nb_t, mk_t, dt)

    def _pair_frequency(self, nb, mask):
        key = nb.unsqueeze(2) * (self.n_nodes + 1) + nb.unsqueeze(1)
        cnt = (key == key.transpose(1, 2)).sum(dim=2).float()
        return (cnt * mask).clamp(min=1.0)

    def _aggregate(self, ids, timestamps):
        nb, mask, dt = self._side(ids, timestamps)
        x_c = self.memory.emb[ids]
        x_i = self.memory.emb[nb]
        freq = self._pair_frequency(nb, mask)
        seq, alpha = self.aggregator(x_c, x_i, dt, freq, freq.max(), mask)
        M_l, M_g = self.partition(seq, dt, mask)
        fused = self.fusion(seq, M_l, M_g, mask)
        pooled = (fused * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return (pooled, seq, alpha, mask, dt)

    def _score_msg(self, h_s, h_d, h_e, mem_s, mem_d, e):
        msg = torch.cat([h_s, h_d, h_e], dim=-1)
        rs = self.gru(msg, mem_s)
        rd = self.gru(msg, mem_d)
        return torch.sigmoid(self.score_mlp(torch.cat([rs, rd, e], dim=-1)).squeeze(-1))

    def _rollout(self, seq_s, mk_s, dt_s, seq_d, mk_d, dt_d, mem_s, mem_d, h_e, e):
        K = self.n_neighbors
        c = torch.cat([seq_s[:, -K:], seq_d[:, -K:]], dim=1)
        v = torch.cat([mk_s[:, -K:], mk_d[:, -K:]], dim=1)
        dt = torch.cat([dt_s[:, -K:], dt_d[:, -K:]], dim=1)
        order = torch.argsort(dt, dim=1)
        gather = lambda x: torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        c = gather(c)
        v = torch.gather(v, 1, order)
        dt = torch.gather(dt, 1, order)
        is_src = (order < K).float()
        center = torch.where(is_src.unsqueeze(-1) > 0, mem_s.unsqueeze(1).expand(-1, 2 * K, -1), mem_d.unsqueeze(1).expand(-1, 2 * K, -1))
        B, N, D = c.shape
        h_s = torch.zeros(B, D, device=self.device)
        h_d = torch.zeros(B, D, device=self.device)
        cnt_s = torch.zeros(B, 1, device=self.device)
        cnt_d = torch.zeros(B, 1, device=self.device)
        states = [(h_s, h_d)]
        log_pi, ent, acted = ([], [], [])
        for i in range(N):
            cur_s = h_s / cnt_s.clamp(min=1.0)
            cur_d = h_d / cnt_d.clamp(min=1.0)
            h_ctx = torch.zeros_like(torch.cat([cur_s, cur_d], dim=-1)) if self.selection_mode == 'noh' else torch.cat([cur_s, cur_d], dim=-1)
            t_i = torch.cos(dt[:, i].unsqueeze(-1) * self.sel_omega)
            state = torch.cat([c[:, i], center[:, i], t_i, h_ctx], dim=-1).detach()
            p = torch.sigmoid(self.sel(state).squeeze(-1)).clamp(1e-06, 1 - 1e-06)
            with torch.no_grad():
                a = (torch.bernoulli(p.detach()) if self.training else (p.detach() >= 0.5).float()) * v[:, i]
                w = (a * is_src[:, i]).unsqueeze(-1)
                h_s = h_s + w * c[:, i]
                cnt_s = cnt_s + w[:, :1]
                w = (a * (1 - is_src[:, i])).unsqueeze(-1)
                h_d = h_d + w * c[:, i]
                cnt_d = cnt_d + w[:, :1]
                states.append((h_s, h_d))
            acted.append(a)
            log_pi.append((torch.log(p) * a + torch.log(1 - p) * (1 - a)) * v[:, i])
            ent.append(-(p * torch.log(p) + (1 - p) * torch.log(1 - p)) * v[:, i])
        with torch.no_grad():
            L = torch.stack([self._score_msg(hs / cnt_s.clamp(min=1.0), hd / cnt_d.clamp(min=1.0), h_e, mem_s, mem_d, e) for hs, hd in states], 1)
            returns = L[:, 1:] - L[:, :-1]
        a_all = torch.stack(acted, 1)
        a_orig = torch.zeros_like(v).scatter(1, order, a_all)
        return (h_s / cnt_s.clamp(min=1.0), h_d / cnt_d.clamp(min=1.0), torch.stack(log_pi, 1), torch.stack(ent, 1), returns, a_orig)

    def forward(self, src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps):
        src_t = torch.as_tensor(src_idxs, dtype=torch.long, device=self.device)
        dst_t = torch.as_tensor(dst_idxs, dtype=torch.long, device=self.device)
        neg_t = torch.as_tensor(neg_idxs, dtype=torch.long, device=self.device)
        eidx = torch.as_tensor(edge_idxs, dtype=torch.long, device=self.device)
        B = src_t.size(0)
        mem_s_pre = self.memory.emb[src_t].clone()
        mem_d_pre = self.memory.emb[dst_t].clone()
        mem_n_pre = self.memory.emb[neg_t].clone()
        h_e = F.relu(self.W_e(self.edge_feat[eidx]))
        m_s, seq_s, alpha_s, mk_s, dt_s = self._aggregate(src_t, timestamps)
        m_d, seq_d, alpha_d, mk_d, dt_d = self._aggregate(dst_t, timestamps)
        policy_loss = torch.zeros((), device=self.device)
        reward = torch.zeros((), device=self.device)
        if self.selection_mode != 'none':
            agg_s, agg_d, log_pi, ent, returns, a_orig = self._rollout(seq_s, mk_s, dt_s, seq_d, mk_d, dt_d, mem_s_pre, mem_d_pre, h_e, h_e)
            K = self.n_neighbors
            m_sel = []
            for seq, mk, dt, a in ((seq_s, mk_s, dt_s, a_orig[:, :K]), (seq_d, mk_d, dt_d, a_orig[:, K:])):
                keep = torch.ones(seq.size(0), seq.size(1), device=self.device)
                keep[:, -K:] = a
                seq_g = seq * keep.unsqueeze(-1)
                M_l, M_g = self.partition(seq_g, dt, mk)
                fused = self.fusion(seq_g, M_l, M_g, mk)
                m_sel.append((fused * mk.unsqueeze(-1)).sum(1) / mk.sum(1, keepdim=True).clamp(min=1.0))
            m_s, m_d = m_sel
            if self.training:
                adv = returns
                sd = adv.std()
                adv = adv / sd.clamp(min=1e-06) if sd > 1e-06 else adv * 0.0
                adv = adv.clamp(-3.0, 3.0)
                policy_loss = (-(log_pi * adv).sum() - self.beta * ent.sum()) / (B * log_pi.size(1))
            reward = returns.mean().detach()
        msg = torch.cat([m_s, m_d, h_e], dim=-1)
        rep_s = self.gru(msg, mem_s_pre)
        rep_d = self.gru(msg, mem_d_pre)
        rep_n = self.gru(msg, mem_n_pre)
        pos = torch.sigmoid(self.score_mlp(torch.cat([rep_s, rep_d, h_e], dim=-1)).squeeze(-1))
        neg = torch.sigmoid(self.score_mlp(torch.cat([rep_s, rep_n, h_e], dim=-1)).squeeze(-1))
        with torch.no_grad():
            self.memory.emb[src_t] = rep_s.detach()
            self.memory.emb[dst_t] = rep_d.detach()
        return (pos, neg, policy_loss, reward)

    def compute_score(self, src_idxs, dst_idxs, neg_idxs, timestamps=None):
        pos = torch.sigmoid((self.memory.emb[src_idxs] * self.memory.emb[dst_idxs]).sum(-1))
        neg = torch.sigmoid((self.memory.emb[src_idxs] * self.memory.emb[neg_idxs]).sum(-1))
        return (pos, neg)

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
