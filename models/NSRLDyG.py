import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .memory import Memory
from torch.distributions import Bernoulli
from torch.autograd import Variable
from torch.nn import MSELoss
_DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class BlockTransformer(nn.Module):

    def __init__(self, embedder: nn.Module, block_decoder: nn.Module, token_decoder: nn.Module, use_token_loss=False, use_block_loss=False, block_length=4):
        super().__init__()
        self.embedder = embedder
        self.block_decoder = block_decoder
        self.token_decoder = token_decoder
        self.block_length = block_length
        self.n_embedding_tokens = embedder.n_embedding_tokens
        self.use_token_loss = use_token_loss
        self.use_block_loss = use_block_loss
        self.config = token_decoder.config

    def forward(self, input_ids, attention_mask, block_attention_mask, labels=None):
        batch_size, n_blocks, block_len = input_ids.shape
        input_embeds = self.embedder(input_ids.view(-1, block_len), attention_mask.view(-1, block_len))
        input_embeds = input_embeds.view(batch_size, -1, input_embeds.shape[-1])
        block_output = self.block_decoder(inputs_embeds=input_embeds, block_attention_mask=block_attention_mask, return_dict=True)
        loss = None
        if self.use_block_loss and labels is not None:
            loss = block_output.loss
        shifted_input_ids = input_ids[:, 1:].reshape(-1, block_len)
        shifted_mask = attention_mask[:, 1:].reshape(-1, block_len)
        block_embeddings = block_output.hidden_states[:, :-self.n_embedding_tokens]
        block_embeddings = block_embeddings.reshape(-1, self.n_embedding_tokens, block_embeddings.shape[-1])
        token_output = self.token_decoder(input_ids=shifted_input_ids, attention_mask=shifted_mask, block_embeddings=block_embeddings, labels=labels[:, 1:].reshape(-1, block_len) if self.use_token_loss else None)
        if self.use_token_loss:
            loss = token_output.loss if loss is None else loss + token_output.loss
        return {'logits': token_output.logits, 'loss': loss}

    def generate(self, input_ids, attention_mask=None, max_length=100):
        if input_ids.dim() == 2:
            batch_size, seq_len = input_ids.shape
            pad_len = (self.block_length - seq_len % self.block_length) % self.block_length
            input_ids = torch.nn.functional.pad(input_ids, (pad_len, 0), value=self.config.pad_token_id)
            input_ids = input_ids.view(batch_size, -1, self.block_length)
        block_embeddings = self.embedder(input_ids.view(-1, self.block_length))
        block_embeddings = block_embeddings.view(input_ids.shape[0], -1, block_embeddings.shape[-1])
        while input_ids.shape[1] * input_ids.shape[2] < max_length:
            block_output = self.block_decoder(inputs_embeds=block_embeddings)
            next_block = block_output.hidden_states[:, -1:]
            token_logits = self.token_decoder.generate(next_block)
            next_token = token_logits.argmax(dim=-1)[..., -1:]
            new_embed = self.embedder(next_token)
            block_embeddings = torch.cat([block_embeddings, new_embed], dim=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        return input_ids

class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, num_heads, hidden_dim, device):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, 'Hidden dimension must be divisible by the number of heads.'
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.device = device
        self.W_p = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_heads)]).to(self.device)
        self.sigma2 = nn.Parameter(torch.ones(1)).to(self.device)
        self.phi = nn.ReLU().to(self.device)
        self.W_k = nn.Linear(hidden_dim, 64).to(self.device)

    def forward(self, x, time_diffs, context_vectors):
        x_expanded = x.unsqueeze(1).expand(-1, 20, -1).to(self.device)
        batch_size, seq_length, _ = x_expanded.shape
        assert time_diffs.shape == (batch_size, seq_length), 'Time diffs shape does not match the sequence length.'
        repeat_times = seq_length // context_vectors.size(1) + (1 if seq_length % context_vectors.size(1) else 0)
        repeated_context = context_vectors.repeat(1, repeat_times, 1)[:, :seq_length, :]
        all_head_outputs = []
        for k in range(self.num_heads):
            projected_x = self.W_p[k](x_expanded).to(self.device)
            time_diffs_broadcast = time_diffs.unsqueeze(-1).to(self.device)
            combined = time_diffs_broadcast * projected_x
            transformed = self.phi(combined).float()
            attn_scores = torch.einsum('bik,bjk->bij', transformed, repeated_context[:, :, :self.head_dim]) * self.sigma2
            attn_weights = F.softmax(attn_scores, dim=2)
            head_output = torch.einsum('bij,bjk->bik', attn_weights, projected_x)
            all_head_outputs.append(head_output.unsqueeze(-2))
        combined_output = torch.cat(all_head_outputs, dim=-2).reshape(batch_size, seq_length, -1)
        output = self.W_k(combined_output)
        return output

class CustomGRU(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(CustomGRU, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_size = hidden_size
        self.update_gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.reset_gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.candidate_hidden_proj = nn.Linear(hidden_size + input_size, hidden_size, bias=True)

    def forward(self, x, h_prev):
        x_proj = self.input_proj(x)
        z_t = torch.sigmoid(self.update_gate_proj(h_prev))
        r_t = torch.sigmoid(self.reset_gate_proj(h_prev))
        h_prev_reset = r_t * h_prev
        combined_input = torch.cat((h_prev_reset, x_proj), dim=1)
        h_tilde = torch.tanh(self.candidate_hidden_proj(combined_input))
        h_t = (1 - z_t) * h_tilde + z_t * h_prev
        return h_t

class ResidualMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ResidualMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList((nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)))
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        original_input = x
        for layer in self.layers:
            x = F.relu(layer(original_input) + original_input if original_input is not x else layer(x))
            original_input = x
        x = self.final_layer(x)
        return F.relu(x)

class TimeEncode(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TimeEncode, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = nn.Linear(input_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        diag_values = 1.0 / 10.0 ** np.arange(self.output_dim, dtype=np.float64)
        weight_matrix = np.diag(diag_values).astype(np.float32)
        self.w.weight = nn.Parameter(torch.from_numpy(weight_matrix).reshape(self.input_dim, self.output_dim))
        self.w.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t))
        return output

class MultiScaleTemporalAttention(nn.Module):

    def __init__(self, num_scales, embed_dim, output_dim, n_out=40):
        super().__init__()
        self.num_scales = num_scales
        self.n_out = n_out
        self.output_dim = output_dim
        self.W_g = nn.ModuleList([nn.Linear(2 * embed_dim, output_dim) for _ in range(num_scales)]).to(_DEV)
        self.decay_rates = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_scales)])
        self.final_linear = nn.Linear(num_scales * output_dim * 2, output_dim * n_out).to(_DEV)
        self.mlp = ResidualMLP(embed_dim * 2, 128, output_dim, 3).to(_DEV)

    def forward(self, neighbors, center_emb, current_time, edge_times, return_weights=False):
        B, N, E = neighbors.shape
        multi_scale_messages = []
        weight_list = []
        n_nb = neighbors.size(1)
        time_encode = TimeEncode(n_nb, n_nb).to(_DEV)
        b = 1
        for scale in range(self.num_scales):
            time_diff = current_time - edge_times
            damp = torch.exp(-self.decay_rates[scale].clamp(min=1e-15).detach().cpu() * time_diff ** 2 / (2 * pow(b, 2))).to(_DEV)
            damp = damp.detach().cpu()
            damp = time_encode(torch.tensor(np.float32(damp)).to(_DEV))
            concat_input = torch.cat([center_emb, neighbors], dim=2)
            transformed = self.W_g[scale](concat_input)
            att_scores = torch.sum(transformed, dim=2)
            att_scores = att_scores * damp
            att_weights = F.softmax(att_scores, dim=1)
            weighted_neighbors = neighbors * att_weights.unsqueeze(-1)
            agg_neighbors = torch.sum(weighted_neighbors, dim=1)
            multi_scale_messages.append(agg_neighbors)
            weight_list.append(att_weights)
        fused_message = torch.cat(multi_scale_messages, dim=1)
        final_output = self.final_linear(fused_message.float())
        final_output = final_output.view(B, self.n_out, self.output_dim)
        if return_weights:
            return (final_output, torch.stack(weight_list, dim=1))
        return final_output

class EmbeddingCombiner(nn.Module):

    def __init__(self, input_size=64, hidden_size=64):
        super(EmbeddingCombiner, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def forward(self, prev_dst, current_dst):
        prev_dst = prev_dst.unsqueeze(0)
        current_dst = current_dst.unsqueeze(0)
        device = _DEV if torch.cuda.is_available() else 'cpu'
        h0 = torch.zeros(1, prev_dst.size(1), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(1, prev_dst.size(1), self.lstm.hidden_size).to(device)
        output, (hn, cn) = self.lstm(current_dst, (h0, c0))
        combined_embedding = output.squeeze(0)
        return combined_embedding

class NSRLDyG(nn.Module):

    def __init__(self, n_nodes, n_neighbors, n_update_neighbors, edge_dim, emb_dim, message_dim, neighbor_finder, edge_feat, tot_time, device='cpu'):
        super(NSRLDyG, self).__init__()
        self.damp_list = []
        self.n_nodes = n_nodes
        self.feat_dim = 100
        self.edge_dim = edge_dim
        self.emb_dim = emb_dim
        self.message_dim = message_dim
        self.edge_feat = torch.tensor(edge_feat, dtype=torch.float).to(device)
        self.memory = Memory(self.n_nodes, self.emb_dim, device)
        self.tot_time = tot_time
        self.n_neighbors = n_neighbors
        self.n_update_neighbors = n_update_neighbors
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.temperature = 5
        self.eps = 1e-10
        self.gamma = 1
        self.W_g = nn.Parameter(torch.zeros((self.emb_dim, self.message_dim // 2)).to(self.device))
        self.a = nn.Parameter(torch.zeros(self.message_dim // 2).to(self.device))
        self.dropout = nn.Dropout(p=0.5)
        nn.init.xavier_normal_(self.W_g)
        self.W_e = nn.Parameter(torch.zeros((self.edge_dim, self.message_dim)).to(self.device))
        nn.init.xavier_normal_(self.W_e)
        self.W_uc = nn.Parameter(torch.zeros((self.emb_dim + 2 * self.message_dim, self.emb_dim)).to(self.device))
        self.W_un = nn.Parameter(torch.zeros((self.emb_dim + self.message_dim, self.emb_dim)).to(self.device))
        nn.init.xavier_normal_(self.W_uc)
        nn.init.xavier_normal_(self.W_un)
        self.W_p = nn.Parameter(torch.zeros((2 * self.message_dim, self.message_dim)).to(self.device))
        nn.init.xavier_normal_(self.W_p)
        self.W_1 = nn.Parameter(torch.zeros((self.emb_dim * 2, self.emb_dim)).to(self.device))
        self.W_2 = nn.Parameter(torch.zeros((self.emb_dim, 1)).to(self.device))
        nn.init.xavier_normal_(self.W_1)
        nn.init.xavier_normal_(self.W_2)
        self.beta = 0.1
        self.use_rns = True
        self.selection_mode = 'parallel'
        d, dm = (self.emb_dim, self.message_dim)
        self.sel_Wg = nn.Parameter(torch.zeros((2 * d, d)).to(self.device))
        self.sel_a = nn.Parameter(torch.zeros(d).to(self.device))
        self.sel_Wm = nn.Parameter(torch.zeros((d, dm // 2)).to(self.device))
        self.sel_Wq = nn.Parameter(torch.zeros((dm // 2 * 2 + dm + d, d)).to(self.device))
        self.sel_W1 = nn.Parameter(torch.zeros((4 * d, d)).to(self.device))
        self.sel_W2 = nn.Parameter(torch.zeros((d, 1)).to(self.device))
        for _p in (self.sel_Wg, self.sel_Wm, self.sel_Wq, self.sel_W1, self.sel_W2):
            nn.init.xavier_normal_(_p)
        self.register_buffer('sel_omega', 1.0 / 10.0 ** (torch.arange(d).float() / d))

    def _lp_mdp(self, h_s, h_d, h_e, mem_s, mem_d, target):
        msg = torch.cat([h_s, h_d, h_e], dim=-1)
        rs = torch.tanh(torch.cat([mem_s, msg], dim=-1) @ self.W_uc)
        rd = torch.tanh(torch.cat([mem_d, msg], dim=-1) @ self.W_uc)
        score = torch.sigmoid((rs * rd).sum(dim=-1))
        return F.binary_cross_entropy(score, target, reduction='none')

    def _forward_mdp(self, src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps):
        dev = self.device
        B = len(src_idxs)
        N, K = (self.n_neighbors, self.n_update_neighbors)
        src_t = torch.as_tensor(src_idxs, dtype=torch.long, device=dev)
        dst_t = torch.as_tensor(dst_idxs, dtype=torch.long, device=dev)
        neg_t = torch.as_tensor(neg_idxs, dtype=torch.long, device=dev)
        eidx = torch.as_tensor(edge_idxs, dtype=torch.long, device=dev)
        mem_s_pre = self.memory.emb[src_t].clone()
        mem_d_pre = self.memory.emb[dst_t].clone()
        mem_n_pre = self.memory.emb[neg_t].clone()
        h_e = torch.relu(self.edge_feat[eidx] @ self.W_e)
        sides = {}
        for tag, ids in (('s', src_t), ('d', dst_t)):
            nb, _, et, mk = self.neighbor_finder.get_temporal_neighbor(ids, timestamps, N, return_mask=True)
            nb_t = torch.as_tensor(nb, dtype=torch.long, device=dev)
            mk_t = torch.as_tensor(mk, dtype=torch.float32, device=dev)
            cur = np.repeat(np.expand_dims(timestamps, 1), N, 1)
            dt = torch.as_tensor((cur - et) / self.tot_time, dtype=torch.float32, device=dev)
            x_i = self.memory.emb[nb_t]
            x_c = self.memory.emb[ids]
            msct = MultiScaleTemporalAttention(3, self.emb_dim, self.emb_dim // 2, n_out=2 * self.n_neighbors)
            center_emb = torch.matmul(x_c, self.W_g).unsqueeze(1).repeat(1, N, 2)
            _, att = msct(x_i, center_emb, cur, et, return_weights=True)
            alpha = att.mean(dim=1) * mk_t
            c = alpha.unsqueeze(-1) * (x_i @ self.sel_Wm)
            sides[tag] = dict(nb=nb_t, dt=dt, valid=mk_t, c=c, m_raw=torch.relu(c.sum(dim=1)))
        msg_ctx = torch.cat([sides['s']['m_raw'], sides['d']['m_raw'], h_e], dim=-1)
        cand_c = torch.cat([sides['s']['c'][:, -K:], sides['d']['c'][:, -K:]], dim=1)
        cand_x = torch.cat([self.memory.emb[sides['s']['nb']][:, -K:], self.memory.emb[sides['d']['nb']][:, -K:]], dim=1)
        cand_dt = torch.cat([sides['s']['dt'][:, -K:], sides['d']['dt'][:, -K:]], dim=1)
        cand_v = torch.cat([sides['s']['valid'][:, -K:], sides['d']['valid'][:, -K:]], dim=1)
        order = torch.argsort(cand_dt, dim=1)
        gather = lambda t: torch.gather(t, 1, order.unsqueeze(-1).expand(-1, -1, t.size(-1)))
        cand_c, cand_x = (gather(cand_c), gather(cand_x))
        cand_dt = torch.gather(cand_dt, 1, order)
        cand_v = torch.gather(cand_v, 1, order)
        is_src = (order < K).float()
        q = torch.relu(torch.cat([msg_ctx.unsqueeze(1).expand(-1, 2 * K, -1), cand_x], dim=-1) @ self.sel_Wq)
        t_feat = torch.cos(cand_dt.unsqueeze(-1) * self.sel_omega)
        half = self.message_dim // 2
        h_s = torch.zeros(B, half, device=dev)
        h_d = torch.zeros(B, half, device=dev)
        pairs = [(h_s, h_d)]
        log_pi, ent = ([], [])
        with torch.no_grad():
            h_s_n, h_d_n = (h_s.clone(), h_d.clone())
        for i in range(2 * K):
            h_ctx = torch.cat([h_s, h_d], dim=-1)
            if self.selection_mode == 'mdp_noh':
                h_ctx = torch.zeros_like(h_ctx).detach()
            s_i = torch.cat([cand_x[:, i], q[:, i], t_feat[:, i], h_ctx], dim=-1).detach()
            p_i = torch.sigmoid(torch.relu(s_i @ self.sel_W1) @ self.sel_W2).squeeze(-1)
            p_i = p_i.clamp(1e-06, 1 - 1e-06)
            with torch.no_grad():
                a_i = (Bernoulli(p_i.detach()).sample() if self.training else (p_i.detach() >= 0.5).float()) * cand_v[:, i]
                h_s = h_s + (a_i * is_src[:, i]).unsqueeze(-1) * cand_c[:, i]
                h_d = h_d + (a_i * (1 - is_src[:, i])).unsqueeze(-1) * cand_c[:, i]
                pairs.append((h_s.clone(), h_d.clone()))
            log_pi.append(Bernoulli(p_i).log_prob(a_i) * cand_v[:, i])
            ent.append(-(p_i * torch.log(p_i) + (1 - p_i) * torch.log(1 - p_i)) * cand_v[:, i])
        one = torch.ones(B, device=dev)
        with torch.no_grad():
            L_pre = torch.stack([self._lp_mdp(hs, hd, h_e, mem_s_pre, mem_d_pre, one) for hs, hd in pairs[:-1]], dim=1)
            L_sel = self._lp_mdp(pairs[-1][0], pairs[-1][1], h_e, mem_s_pre, mem_d_pre, one)
            returns = L_pre - L_sel.unsqueeze(1)
        if self.training:
            adv = returns
            sd = adv.std()
            adv = adv / sd.clamp(min=1e-06) if sd > 1e-06 else adv * 0.0
            adv = adv.clamp(-2.0, 2.0)
            policy_loss = (-(torch.stack(log_pi, 1) * adv).sum() - self.beta * torch.stack(ent, 1).sum()) / B
        else:
            policy_loss = torch.zeros((), device=dev)
        msg_sel = torch.cat([h_s, h_d, h_e], dim=-1)
        rs = torch.tanh(torch.cat([mem_s_pre, msg_sel], dim=-1) @ self.W_uc)
        rd = torch.tanh(torch.cat([mem_d_pre, msg_sel], dim=-1) @ self.W_uc)
        rn = torch.tanh(torch.cat([mem_n_pre, msg_sel], dim=-1) @ self.W_uc)
        pos_score = torch.sigmoid((rs * rd).sum(dim=-1))
        neg_score = torch.sigmoid((rs * rn).sum(dim=-1))
        with torch.no_grad():
            self.memory.emb[src_t] = rs.detach()
            self.memory.emb[dst_t] = rd.detach()
        return (pos_score, neg_score, policy_loss, returns.mean().detach())

    def forward(self, src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps):
        if self.selection_mode != 'parallel':
            return self._forward_mdp(src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps)
        custom_gru = CustomGRU(input_size=32, hidden_size=32)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        custom_gru = custom_gru.to(device)
        MultiScaleTemporalAttention_model = MultiScaleTemporalAttention(3, self.emb_dim, self.emb_dim // 2, n_out=2 * self.n_neighbors)
        message = []
        model_EmbeddingCombiner = EmbeddingCombiner().to(_DEV)
        for i, idxs in enumerate([src_idxs, dst_idxs]):
            neighbors, _, edge_times = self.neighbor_finder.get_temporal_neighbor(idxs, timestamps, self.n_neighbors)
            current_time = np.repeat(np.expand_dims(timestamps, axis=1), self.n_neighbors, axis=1)
            interval = (current_time - edge_times) / self.tot_time
            damp = 1 / (1 + self.gamma * interval)
            neighbors = torch.from_numpy(neighbors).long().to(self.device)
            bs = neighbors.shape[0]
            neighbor_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_neighbors, self.emb_dim)
            damp = np.repeat(np.expand_dims(damp, axis=2), self.emb_dim, axis=2)
            center_emb = torch.matmul(self.memory.emb[idxs], self.W_g).unsqueeze(dim=1).repeat(1, self.n_neighbors, 2)
            h = MultiScaleTemporalAttention_model(neighbor_emb, center_emb, current_time, edge_times)
            center_emb_prev = torch.matmul(self.memory.emb[idxs], self.W_g).to(self.device)
            h = custom_gru(h[:, i, :], center_emb_prev)
            h = h.unsqueeze(dim=1).repeat(1, 2 * self.n_neighbors, 1)
            h = h + torch.matmul(self.memory.emb[idxs], self.W_g).unsqueeze(dim=1).repeat(1, 2 * self.n_neighbors, 1)
            message.append(h)
        prev_src = self.memory.emb[src_idxs].to(_DEV)
        h = torch.cat((message[0], message[1]), dim=2).sum(dim=1).relu()
        h_e = torch.matmul(self.edge_feat[edge_idxs], self.W_e).relu()
        h = torch.cat((h, h_e), dim=1)
        to_updated_src = torch.matmul(torch.cat((self.memory.emb[src_idxs], h), dim=1), self.W_uc).tanh().to(_DEV)
        to_updated_dst = torch.matmul(torch.cat((self.memory.emb[dst_idxs], h), dim=1), self.W_uc).tanh().to(_DEV)
        prev_dst = self.memory.emb[dst_idxs].to(_DEV)
        self.memory.emb[src_idxs] = to_updated_src
        self.memory.emb[dst_idxs] = to_updated_dst
        model = MultiHeadAttention(input_dim=128, num_heads=16, hidden_dim=128, device=_DEV)
        for idxs in [src_idxs, dst_idxs]:
            neighbors, _, edge_times = self.neighbor_finder.get_temporal_neighbor(idxs, timestamps, self.n_update_neighbors)
            current_time = np.repeat(np.expand_dims(timestamps, axis=1), self.n_update_neighbors, axis=1)
            interval = (current_time - edge_times) / self.tot_time
            decay_rates = torch.linspace(0.1, 2.0, len(neighbors))
            decayed_intervals = [torch.exp(-decay_rate * interval) for decay_rate in decay_rates]
            neighbors = torch.from_numpy(neighbors).long().to(self.device)
            bs = neighbors.shape[0]
            neighbor_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_update_neighbors, self.emb_dim)
            damp = 1 / (1 + self.gamma * interval)
            damp = np.repeat(np.expand_dims(damp, axis=2), self.emb_dim, axis=2)
            neighbor_emb = neighbor_emb * torch.from_numpy(np.float32(damp)).to(self.device)
            damp1 = 1 / (1 + self.gamma * interval)
            damp1 = torch.tensor(damp1)
            changed_emb = model(h, damp1, neighbor_emb)
            x = torch.cat((neighbor_emb, changed_emb), dim=2)
            x.detach_()
            x = torch.matmul(x, self.W_1)
            x = x.relu()
            x = torch.matmul(x, self.W_2)
            probs = x.sigmoid()
            changed_emb = torch.matmul(torch.cat((self.memory.emb[neighbors.flatten()], changed_emb.flatten().view(-1, self.emb_dim)), dim=1), self.W_un).tanh()
            policy_loss = 0
            reward = 0
            policy_map = probs.data.clone()
            policy_map[policy_map < 0.5] = 0.0
            policy_map[policy_map >= 0.5] = 1.0
            policy_map = Variable(policy_map)
            distr = Bernoulli(probs)
            policy = distr.sample()
            if not self.training:
                if self.use_rns:
                    mask = policy_map.repeat(1, 1, self.emb_dim).flatten().view(-1, self.emb_dim)
                    self.memory.emb[neighbors.flatten()] = mask * changed_emb + (1 - mask) * self.memory.emb[neighbors.flatten()]
                else:
                    self.memory.emb[neighbors.flatten()] = changed_emb
            else:
                policy_map = policy_map.repeat(1, 1, self.emb_dim).flatten().view(-1, self.emb_dim)
                policy_sample = policy.repeat(1, 1, self.emb_dim).flatten().view(-1, self.emb_dim)
                ori_emb = self.memory.emb[neighbors.flatten()]
                updated_emb_map = policy_map * changed_emb + (1 - policy_map) * ori_emb
                updated_emb_sample = policy_sample * changed_emb + (1 - policy_sample) * ori_emb
                reward_map = self.get_reward(idxs, updated_emb_map).detach()
                reward_sample = self.get_reward(idxs, updated_emb_sample).detach()
                advantage = reward_sample - reward_map
                loss = -distr.log_prob(policy) * Variable(advantage).expand_as(policy)
                loss = loss.sum()
                probs = probs.clamp(1e-15, 1 - 1e-15)
                entropy_loss = -probs * torch.log(probs)
                entropy_loss = self.beta * entropy_loss.sum()
                loss = (loss - entropy_loss) / bs / self.n_update_neighbors
                self.memory.emb[neighbors.flatten()] = updated_emb_sample
                policy_loss += loss
                reward += reward_sample
        pos_score, neg_score = self.compute_score(src_idxs, dst_idxs, neg_idxs)
        return (pos_score, neg_score, policy_loss, reward)

    def get_reward(self, idxs, neighbor_emb):
        reg_strength = 0.01
        central_emb = self.memory.emb[idxs].repeat(1, self.n_update_neighbors).view(-1, self.emb_dim)
        central_emb_norm = F.normalize(central_emb, p=2, dim=1).detach()
        neighbor_emb_norm = F.normalize(neighbor_emb, p=2, dim=1)
        cos_sim = torch.matmul(central_emb_norm, neighbor_emb_norm.t())
        reg_term = torch.norm(central_emb, p=2, dim=1).mean()
        reward = cos_sim.mean() - reg_strength * reg_term
        return reward

    def compute_sim(self, src_idxs):
        src_norm = F.normalize(self.memory.emb[src_idxs], p=2, dim=1)
        emb_norm = F.normalize(self.memory.emb, p=2, dim=1)
        cos_sim = torch.matmul(src_norm, emb_norm.t())
        sorted_cos_sim, idx = cos_sim.sort(descending=True)
        return (sorted_cos_sim, idx)

    def compute_score(self, src_idxs, dst_idxs, neg_idxs):
        pos_score = torch.sum(self.memory.emb[src_idxs] * self.memory.emb[dst_idxs], dim=1)
        neg_score = torch.sum(self.memory.emb[src_idxs] * self.memory.emb[neg_idxs], dim=1)
        return (pos_score.sigmoid(), neg_score.sigmoid())

    def reset_graph(self):
        self.memory.reset()

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder

    def detach_memory(self):
        self.memory.detach_memory()

    def back_up_memory(self):
        return self.memory.emb.clone()

    def restore_memory(self, back_up):
        self.memory.emb = nn.Parameter(back_up.detach().clone(), requires_grad=False)
