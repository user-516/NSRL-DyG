import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .memory import Memory
from torch.distributions import Bernoulli
from torch.autograd import Variable


from torch.nn import MSELoss

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
class BlockTransformer(nn.Module):
    def __init__(
        self,
        embedder: nn.Module,
        block_decoder: nn.Module,
        token_decoder: nn.Module,
        use_token_loss=False,
        use_block_loss=False,
        block_length=4
    ):
        super().__init__()
        self.embedder = embedder
        self.block_decoder = block_decoder
        self.token_decoder = token_decoder
        self.block_length = block_length
        self.n_embedding_tokens = embedder.n_embedding_tokens
        
        # Loss flags
        self.use_token_loss = use_token_loss
        self.use_block_loss = use_block_loss
        
        # For compatibility
        self.config = token_decoder.config

    def forward(self, input_ids, attention_mask, block_attention_mask, labels=None):
        batch_size, n_blocks, block_len = input_ids.shape
        
        # 1. Embedding Layer
        input_embeds = self.embedder(
            input_ids.view(-1, block_len),
            attention_mask.view(-1, block_len)
        )
        
        # 2. Block Decoding
        input_embeds = input_embeds.view(batch_size, -1, input_embeds.shape[-1])
        block_output = self.block_decoder(
            inputs_embeds=input_embeds, 
            block_attention_mask=block_attention_mask,
            return_dict=True
        )
        
        loss = None
        # Block loss calculation
        if self.use_block_loss and labels is not None:
            loss = block_output.loss
            
        # 3. Token Decoding
        # Shift blocks for token prediction
        shifted_input_ids = input_ids[:, 1:].reshape(-1, block_len)
        shifted_mask = attention_mask[:, 1:].reshape(-1, block_len)
        
        # Get block embeddings (excluding last block)
        block_embeddings = block_output.hidden_states[:, :-self.n_embedding_tokens]
        block_embeddings = block_embeddings.reshape(-1, self.n_embedding_tokens, block_embeddings.shape[-1])
        
        # Token decoder forward pass
        token_output = self.token_decoder(
            input_ids=shifted_input_ids,
            attention_mask=shifted_mask,
            block_embeddings=block_embeddings,
            labels=labels[:, 1:].reshape(-1, block_len) if self.use_token_loss else None
        )
        
        # Combine losses
        if self.use_token_loss:
            loss = token_output.loss if loss is None else loss + token_output.loss
            
        return {
            "logits": token_output.logits,
            "loss": loss
        }

    def generate(self, input_ids, attention_mask=None, max_length=100):
        # Convert to block format if needed
        if input_ids.dim() == 2:
            batch_size, seq_len = input_ids.shape
            pad_len = (self.block_length - seq_len % self.block_length) % self.block_length
            input_ids = torch.nn.functional.pad(input_ids, (pad_len, 0), value=self.config.pad_token_id)
            input_ids = input_ids.view(batch_size, -1, self.block_length)
        
        # Initial embedding
        block_embeddings = self.embedder(input_ids.view(-1, self.block_length))
        block_embeddings = block_embeddings.view(input_ids.shape[0], -1, block_embeddings.shape[-1])
        
        # Generation loop
        while input_ids.shape[1] * input_ids.shape[2] < max_length:
            # Block decoding
            block_output = self.block_decoder(inputs_embeds=block_embeddings)
            
            # Predict next block
            next_block = block_output.hidden_states[:, -1:]
            token_logits = self.token_decoder.generate(next_block)
            
            # Get next token from logits
            next_token = token_logits.argmax(dim=-1)[..., -1:]
            
            # Update block embeddings
            new_embed = self.embedder(next_token)
            block_embeddings = torch.cat([block_embeddings, new_embed], dim=1)
            
            # Update input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            
        return input_ids

class MultiHeadAttention(nn.Module):
	def __init__(self, input_dim, num_heads, hidden_dim,device):
		super(MultiHeadAttention, self).__init__()
		assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads."
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.head_dim = hidden_dim // num_heads
		self.device = device
		# 为每个头创建一个线性层
		self.W_p = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_heads)]).to(self.device)

		# 可学习的缩放因子
		self.sigma2 = nn.Parameter(torch.ones(1)).to(self.device)

		# 激活函数ReLU
		self.phi = nn.ReLU().to(self.device)

		# 线性层用于输出变换，调整以适应最终的输出维度
		self.W_k = nn.Linear(hidden_dim, 64).to(self.device)  # 输出维度调整为64以匹配需求

	def forward(self, x, time_diffs, context_vectors):
		# 将x视为单个时间点的特征，我们扩展它以模拟序列
		x_expanded = x.unsqueeze(1).expand(-1, 20, -1).to(self.device) # 扩展到n_update_degree个时间步以匹配context_vectors
		batch_size, seq_length, _ = x_expanded.shape

		# 确保时间差信息维度与序列长度匹配
		assert time_diffs.shape == (batch_size, seq_length), "Time diffs shape does not match the sequence length."

		# 重复context_vectors以匹配seq_length
		repeat_times = seq_length // context_vectors.size(1) + (1 if seq_length % context_vectors.size(1) else 0)
		repeated_context = context_vectors.repeat(1, repeat_times, 1)[:, :seq_length, :]  # 确保最终长度不超过seq_length

		# 初始化用于存储最终输出的列表
		all_head_outputs = []

		for k in range(self.num_heads):
			# 应用线性变换到扩展后的输入特征
			projected_x = self.W_p[k](x_expanded).to(self.device)

			# 将时间差信息与输入特征结合
			time_diffs_broadcast = time_diffs.unsqueeze(-1).to(self.device)
			combined = time_diffs_broadcast * projected_x

			# 应用非线性变换
			transformed = self.phi(combined).float()

			# 计算注意力分数
			attn_scores = torch.einsum('bik,bjk->bij', transformed,
									   repeated_context[:, :, :self.head_dim]) * self.sigma2

			# 归一化得到注意力权重
			attn_weights = F.softmax(attn_scores, dim=2)

			# 应用注意力权重到输入特征
			head_output = torch.einsum('bij,bjk->bik', attn_weights, projected_x)

			# 存储每个头的输出
			all_head_outputs.append(head_output.unsqueeze(-2))

		# 合并所有头的输出
		combined_output = torch.cat(all_head_outputs, dim=-2).reshape(batch_size, seq_length, -1)

		# 应用最后的线性变换以获得所需的输出维度
		output = self.W_k(combined_output)
		return output



class CustomGRU(nn.Module):
    """
    Custom implementation of a Gated Recurrent Unit (GRU) cell.

    Parameters:
    - input_size: int, the number of expected features in the input `x`
    - hidden_size: int, the number of features in the hidden state `h`
    """

    def __init__(self, input_size, hidden_size):
        super(CustomGRU, self).__init__()
        # Define the linear transformations for the gates
        self.input_proj = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_size = hidden_size
        self.update_gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.reset_gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.candidate_hidden_proj = nn.Linear(hidden_size + input_size, hidden_size, bias=True)

    def forward(self, x, h_prev):
        """
        Forward pass of the CustomGRU cell.

        Parameters:
        - x: Tensor, input tensor of shape (batch_size, input_size)
        - h_prev: Tensor, previous hidden state of shape (batch_size, hidden_size)

        Returns:
        - h_t: Tensor, current hidden state
        """

        # Project input to hidden size
        x_proj = self.input_proj(x)

        # Compute update and reset gates
        z_t = torch.sigmoid(self.update_gate_proj(h_prev))
        r_t = torch.sigmoid(self.reset_gate_proj(h_prev))

        # Apply reset gate to the previous hidden state
        h_prev_reset = r_t * h_prev

        # Compute candidate hidden state
        combined_input = torch.cat((h_prev_reset, x_proj), dim=1)
        h_tilde = torch.tanh(self.candidate_hidden_proj(combined_input))

        # Update current hidden state
        h_t = (1 - z_t) * h_tilde + z_t * h_prev

        return h_t
	



class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ResidualMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 创建多层全连接层
        self.layers = nn.ModuleList(
            nn.Linear((input_dim if i == 0 else hidden_dim), hidden_dim) for i in range(num_layers)
        )
        
        # 最后的全连接层从隐藏层维度映射到输出维度
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 保存输入用于残差连接
        original_input = x
        # 通过多层全连接层
        for layer in self.layers:
            x = F.relu(layer(original_input) + original_input if original_input is not x else layer(x))
            original_input = x
        # 最后的全连接层
        x = self.final_layer(x)
        return F.relu(x)



# 时间编码
class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, input_dim, output_dim):
        super(TimeEncode, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = nn.Linear(input_dim, output_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        # 创建一个对角线为1/10^n的矩阵，其中 n 为对角线索引
        # float64 avoids the overflow that 10**j hits for j above ~38; values below the
        # float32 range simply underflow to zero when the matrix is cast.
        diag_values = 1.0 / (10.0 ** np.arange(self.output_dim, dtype=np.float64))
        weight_matrix = np.diag(diag_values).astype(np.float32)
        self.w.weight = nn.Parameter(torch.from_numpy(weight_matrix).reshape(self.input_dim, self.output_dim))
        self.w.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        # 直接应用线性层和余弦函数
        output = torch.cos(self.w(t))
        return output



class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, num_scales, embed_dim, output_dim, n_out=40):
        super().__init__()
        self.num_scales = num_scales
        self.n_out = n_out
        self.output_dim = output_dim
        self.W_g = nn.ModuleList([nn.Linear(2 * embed_dim, output_dim) for _ in range(num_scales)]).to(_DEV) # 修改线性层以拼接后的维度
        self.decay_rates = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_scales)])
        self.final_linear = nn.Linear(num_scales * output_dim * 2, output_dim * n_out).to(_DEV)  # 修改最终线性层输入维度
        self.mlp = ResidualMLP(embed_dim * 2 , 128, output_dim,3).to(_DEV)

		
    def forward(self, neighbors, center_emb, current_time, edge_times,
                return_weights=False):
        B, N, E = neighbors.shape  # E is the embed_dim
        multi_scale_messages = []
        weight_list = []
        # 实例化 TimeEncode 对象
        n_nb = neighbors.size(1)
        time_encode = TimeEncode(n_nb, n_nb).to(_DEV)
        b = 1
        for scale in range(self.num_scales):
            time_diff = current_time - edge_times
            # damp = torch.exp(-self.decay_rates[scale].clamp(min=1e-15).detach().cpu()* time_diff).to(_DEV)  #(16,20)
			# =exp⁡(-(Δt^2)/(2b^2 )),这里高斯带宽取b = 3
            damp = torch.exp(-self.decay_rates[scale].clamp(min=1e-15).detach().cpu()* (time_diff**2)/(2*pow(b,2))).to(_DEV)  #(16,20)
            damp = damp.detach().cpu()
            damp = time_encode(torch.tensor(np.float32(damp)).to(_DEV))
            # Concatenate center_emb and neighbors
            concat_input = torch.cat([center_emb, neighbors], dim=2) #（16，20，128）
            
            # Apply linear transformation
            transformed = self.W_g[scale](concat_input)   # （16，20，32）
            # transformed = self.mlp(concat_input)	
             
            # Calculate attention scores
            att_scores = torch.sum(transformed, dim=2)  # Sum over the embedding dimension (16,20)
            
            # Apply dampening factor
            att_scores = (att_scores * damp)  #(16,20)
			
            
            # Normalize attention scores
            att_weights = F.softmax(att_scores, dim=1) # (16,20)
           
            # Weighted sum of neighbors using attention weights
            weighted_neighbors = neighbors * att_weights.unsqueeze(-1)   #(16,20,64)
            agg_neighbors = torch.sum(weighted_neighbors, dim=1)  # Aggregate over neighbors (16,64)
         
            # Append the transformed and aggregated message for this scale
            multi_scale_messages.append(agg_neighbors)
            weight_list.append(att_weights)
        
        # Concatenate messages from all scales
        fused_message = torch.cat(multi_scale_messages, dim=1)   #(16,64*3)
        final_output = self.final_linear(fused_message.float())
        
        
        # Reshape to (B, 2 * N, output_dim) as we have double the number of elements after concatenation
        # print(final_output.shape)
        final_output = final_output.view(B, self.n_out, self.output_dim)
        if return_weights:
            return final_output, torch.stack(weight_list, dim=1)
        return final_output



class EmbeddingCombiner(nn.Module):
    def __init__(self, input_size=64, hidden_size=64):
        super(EmbeddingCombiner, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def forward(self, prev_dst, current_dst):
        """
        :param prev_dst: 上一时刻的嵌入，维度为 [16, 64]
        :param current_dst: 当前时刻的嵌入，维度为 [16, 64]
        :return: 结合上一时刻和当前时刻的嵌入表示
        """
        # 将输入转换为适合 LSTM 的格式，即添加时间维度
        prev_dst = prev_dst.unsqueeze(0)  
        current_dst = current_dst.unsqueeze(0)  

        # 初始化 LSTM 的隐藏状态和细胞状态为 0
        device = _DEV if torch.cuda.is_available() else 'cpu'
        h0 = torch.zeros(1, prev_dst.size(1), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(1, prev_dst.size(1), self.lstm.hidden_size).to(device)

        # 通过 LSTM 得到输出
        output, (hn, cn) = self.lstm(current_dst, (h0, c0))

        # 取出最后一个时间步的输出作为结果
        combined_embedding = output.squeeze(0)  

        return combined_embedding
    




    

class NSRLDyG(nn.Module):
	def __init__(self, n_nodes, n_neighbors, n_update_neighbors,
				 edge_dim, emb_dim, message_dim,
				 neighbor_finder, edge_feat, tot_time, device='cpu'):
		super(NSRLDyG, self).__init__()
		# No seeding here on purpose: the run seed belongs to the caller.  Seeding
		# inside __init__ would override it and make every run bit-identical,
		# which would render the reported standard deviations meaningless.
		# graph
		self.damp_list = []
		self.n_nodes = n_nodes
		# Dimensions
		self.feat_dim = 100
		self.edge_dim = edge_dim
		self.emb_dim = emb_dim
		self.message_dim = message_dim
		# memory
		self.edge_feat = torch.tensor(edge_feat, dtype=torch.float).to(device)
		self.memory = Memory(self.n_nodes, self.emb_dim, device)
		# other
		self.tot_time = tot_time
		self.n_neighbors = n_neighbors
		self.n_update_neighbors = n_update_neighbors
		self.neighbor_finder = neighbor_finder
		self.device = device
		self.temperature = 5
		self.eps = 1e-10
		self.gamma = 1
		# GAT
		self.W_g = nn.Parameter(torch.zeros((self.emb_dim, self.message_dim // 2)).to(self.device))
		self.a = nn.Parameter(torch.zeros(self.message_dim//2).to(self.device))
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
		# Inference-time switch used by domino_probe.py: when False the gate is bypassed
		# and every candidate neighbour's representation is updated, i.e. the w/o-RNS arm.
		self.use_rns = True
		# ---- Sequential MDP path (selection_mode != 'parallel') -----------------
		# State s_i = x_i || q_i || cos(dt*omega) || h_{<i}, action a_i in {0,1},
		# transition h_{<i+1} = h_{<i} + a_i c_i, reward = marginal drop in L_p.
		self.selection_mode = 'parallel'
		d, dm = self.emb_dim, self.message_dim
		self.sel_Wg = nn.Parameter(torch.zeros((2 * d, d)).to(self.device))
		self.sel_a = nn.Parameter(torch.zeros(d).to(self.device))
		self.sel_Wm = nn.Parameter(torch.zeros((d, dm // 2)).to(self.device))
		self.sel_Wq = nn.Parameter(torch.zeros((dm // 2 * 2 + dm + d, d)).to(self.device))
		self.sel_W1 = nn.Parameter(torch.zeros((4 * d, d)).to(self.device))
		self.sel_W2 = nn.Parameter(torch.zeros((d, 1)).to(self.device))
		for _p in (self.sel_Wg, self.sel_Wm, self.sel_Wq, self.sel_W1, self.sel_W2):
			nn.init.xavier_normal_(_p)
		self.register_buffer('sel_omega', 1.0 / (10.0 ** (torch.arange(d).float() / d)))
		# self.mlp1 = ResidualMLP1(32 , 64, 32,3).to(_DEV)

	
	
	
	def _lp_mdp(self, h_s, h_d, h_e, mem_s, mem_d, target):
		"""Eq (22) on the message assembled from the partial aggregate (h_s, h_d)."""
		msg = torch.cat([h_s, h_d, h_e], dim=-1)
		rs = torch.tanh(torch.cat([mem_s, msg], dim=-1) @ self.W_uc)
		rd = torch.tanh(torch.cat([mem_d, msg], dim=-1) @ self.W_uc)
		score = torch.sigmoid((rs * rd).sum(dim=-1))
		return F.binary_cross_entropy(score, target, reduction='none')

	def _forward_mdp(self, src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps):
		"""Sequential neighbor selection as a finite-horizon MDP.

		State   s_i = x_i || q_i || cos(dt_i * omega) || h_{<i}
		Action  a_i in {0,1}, whether candidate i enters the aggregation
		Trans.  h_{<i+1} = h_{<i} + a_i * c_i
		Reward  r_i = L_p(h_{<i}) - L_p(h_{<i+1})

		Candidates are the top n_update_degree neighbours of each endpoint, merged
		and ordered by recency (smallest dt first).  With selection_mode == 'mdp_noh'
		the running aggregate is withheld from the state, which isolates the effect of
		h_{<i} from the effect of the binary gate itself.
		"""
		dev = self.device
		B = len(src_idxs)
		N, K = self.n_neighbors, self.n_update_neighbors
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
			nb, _, et, mk = self.neighbor_finder.get_temporal_neighbor(
				ids, timestamps, N, return_mask=True)
			nb_t = torch.as_tensor(nb, dtype=torch.long, device=dev)
			mk_t = torch.as_tensor(mk, dtype=torch.float32, device=dev)
			cur = np.repeat(np.expand_dims(timestamps, 1), N, 1)
			dt = torch.as_tensor((cur - et) / self.tot_time, dtype=torch.float32, device=dev)
			x_i = self.memory.emb[nb_t]
			x_c = self.memory.emb[ids]
			# Reuse the TFDGA attention of the parallel path, so that both selection
			# mechanisms share the same front end and differ only in how they gate.
			msct = MultiScaleTemporalAttention(3, self.emb_dim, self.emb_dim // 2,
			                                   n_out=2 * self.n_neighbors)
			center_emb = torch.matmul(x_c, self.W_g).unsqueeze(1).repeat(1, N, 2)
			_, att = msct(x_i, center_emb, cur, et, return_weights=True)
			alpha = att.mean(dim=1) * mk_t
			c = alpha.unsqueeze(-1) * (x_i @ self.sel_Wm)
			sides[tag] = dict(nb=nb_t, dt=dt, valid=mk_t, c=c,
			                  m_raw=torch.relu(c.sum(dim=1)))

		msg_ctx = torch.cat([sides['s']['m_raw'], sides['d']['m_raw'], h_e], dim=-1)

		# candidates: last K per endpoint, merged and sorted by recency
		cand_c = torch.cat([sides['s']['c'][:, -K:], sides['d']['c'][:, -K:]], dim=1)
		cand_x = torch.cat([self.memory.emb[sides['s']['nb']][:, -K:],
		                    self.memory.emb[sides['d']['nb']][:, -K:]], dim=1)
		cand_dt = torch.cat([sides['s']['dt'][:, -K:], sides['d']['dt'][:, -K:]], dim=1)
		cand_v = torch.cat([sides['s']['valid'][:, -K:], sides['d']['valid'][:, -K:]], dim=1)

		order = torch.argsort(cand_dt, dim=1)                     # most recent first
		gather = lambda t: torch.gather(t, 1, order.unsqueeze(-1).expand(-1, -1, t.size(-1)))
		cand_c, cand_x = gather(cand_c), gather(cand_x)
		cand_dt = torch.gather(cand_dt, 1, order)
		cand_v = torch.gather(cand_v, 1, order)
		is_src = (order < K).float()

		q = torch.relu(torch.cat([msg_ctx.unsqueeze(1).expand(-1, 2 * K, -1),
		                          cand_x], dim=-1) @ self.sel_Wq)
		t_feat = torch.cos(cand_dt.unsqueeze(-1) * self.sel_omega)

		half = self.message_dim // 2
		h_s = torch.zeros(B, half, device=dev)
		h_d = torch.zeros(B, half, device=dev)
		pairs = [(h_s, h_d)]
		log_pi, ent = [], []
		with torch.no_grad():
			h_s_n, h_d_n = h_s.clone(), h_d.clone()
		for i in range(2 * K):
			h_ctx = torch.cat([h_s, h_d], dim=-1)
			if self.selection_mode == 'mdp_noh':
				h_ctx = torch.zeros_like(h_ctx).detach()
			s_i = torch.cat([cand_x[:, i], q[:, i], t_feat[:, i], h_ctx], dim=-1).detach()
			p_i = torch.sigmoid(torch.relu(s_i @ self.sel_W1) @ self.sel_W2).squeeze(-1)
			p_i = p_i.clamp(1e-6, 1 - 1e-6)
			with torch.no_grad():
				a_i = (Bernoulli(p_i.detach()).sample() if self.training
				       else (p_i.detach() >= 0.5).float()) * cand_v[:, i]
				h_s = h_s + (a_i * is_src[:, i]).unsqueeze(-1) * cand_c[:, i]
				h_d = h_d + (a_i * (1 - is_src[:, i])).unsqueeze(-1) * cand_c[:, i]
				pairs.append((h_s.clone(), h_d.clone()))
			log_pi.append(Bernoulli(p_i).log_prob(a_i) * cand_v[:, i])
			ent.append(-(p_i * torch.log(p_i) + (1 - p_i) * torch.log(1 - p_i)) * cand_v[:, i])

		one = torch.ones(B, device=dev)
		with torch.no_grad():
			L_pre = torch.stack([self._lp_mdp(hs, hd, h_e, mem_s_pre, mem_d_pre, one)
			                     for hs, hd in pairs[:-1]], dim=1)
			L_sel = self._lp_mdp(pairs[-1][0], pairs[-1][1], h_e, mem_s_pre, mem_d_pre, one)
			returns = L_pre - L_sel.unsqueeze(1)

		if self.training:
			adv = returns
			sd = adv.std()
			adv = adv / sd.clamp(min=1e-6) if sd > 1e-6 else adv * 0.0
			adv = adv.clamp(-2.0, 2.0)
			policy_loss = (-(torch.stack(log_pi, 1) * adv).sum()
			               - self.beta * torch.stack(ent, 1).sum()) / B
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
		return pos_score, neg_score, policy_loss, returns.mean().detach()

	def forward(self, src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps):
		if self.selection_mode != 'parallel':
			return self._forward_mdp(src_idxs, dst_idxs, neg_idxs, edge_idxs, timestamps)
		# GRU模块
		custom_gru = CustomGRU(input_size=32, hidden_size=32)
		# 指定设备
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


		# 将模型移动到指定的设备上
		custom_gru = custom_gru.to(device)

		MultiScaleTemporalAttention_model = MultiScaleTemporalAttention(3, self.emb_dim, self.emb_dim//2, n_out=2*self.n_neighbors)
		message = []
		model_EmbeddingCombiner = EmbeddingCombiner().to(_DEV)
		for i, idxs in enumerate([src_idxs, dst_idxs]):

			# 获取给定节点 idxs 在指定时间戳 timestamps 下的 n_neighbors 个最近邻节点及其对应的边创建时间 edge_times。
			neighbors, _, edge_times = self.neighbor_finder.get_temporal_neighbor(idxs, timestamps, self.n_neighbors)
			# 将输入的时间戳 timestamps 沿第1轴（列）复制 n_neighbors 次，使得每个中心节点对应其所有邻居节点的时间戳,current_time的形状为[len(timestamps),self.n_neighbors],此代码中为[16,20]
			current_time = np.repeat(np.expand_dims(timestamps, axis=1), self.n_neighbors, axis=1)
			#  计算当前时间 current_time 与邻居节点边创建时间 edge_times 之间的时间间隔，并将其除以总时间 self.tot_time 进行标准化。
			interval = (current_time - edge_times) / self.tot_time
			# # (16,20)
			damp = 1 / (1 + self.gamma * interval)
			neighbors = torch.from_numpy(neighbors).long().to(self.device)
			bs = neighbors.shape[0]
			# 从记忆模块 self.memory 中提取邻居节点的嵌入向量。(16,20,64)
			neighbor_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_neighbors, self.emb_dim)

			# (16,20,64)
			# 重复 damp 以匹配 neighbor_emb 的维度，并转换为 float32 类型
			damp = np.repeat(np.expand_dims(damp, axis=2), self.emb_dim, axis=2)
			# damp = torch.tensor(damp, dtype=torch.float32).to(self.device)
			# 邻居节点嵌入乘以时间衰减因子 对于公式（1）中分子的后半部分
			# neighbor_emb = neighbor_emb * torch.from_numpy(np.float32(damp)).to(self.device)
			#(16,20,64)
			center_emb = torch.matmul(self.memory.emb[idxs], self.W_g).unsqueeze(dim=1).repeat(1, self.n_neighbors, 2)
			
			h = MultiScaleTemporalAttention_model(neighbor_emb, center_emb, current_time, edge_times)  #(16,40,32)
			# h = self.mlp1(h)
			center_emb_prev = torch.matmul(self.memory.emb[idxs], self.W_g).to(self.device)
			h = custom_gru(h[:,i,:],center_emb_prev)
			# (16,40,32)
			h = h.unsqueeze(dim=1).repeat(1, 2 * self.n_neighbors, 1)
			h = h + torch.matmul(self.memory.emb[idxs], self.W_g).unsqueeze(dim=1).repeat(1, 2 * self.n_neighbors, 1)
			
			# =========================================================================

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
            
		# self.memory.emb[src_idxs] = model_EmbeddingCombiner(prev_src,to_updated_src)
		# self.memory.emb[dst_idxs] = model_EmbeddingCombiner(prev_dst,to_updated_dst)

		# changed_emb_lt = []
		model = MultiHeadAttention(input_dim=128, num_heads=16, hidden_dim=128,device=_DEV)
		# prop
		for idxs in [src_idxs, dst_idxs]:
			neighbors, _, edge_times = self.neighbor_finder.get_temporal_neighbor(idxs, timestamps,
																				  self.n_update_neighbors)
			current_time = np.repeat(np.expand_dims(timestamps, axis=1), self.n_update_neighbors, axis=1)
			interval = (current_time - edge_times) / self.tot_time
			decay_rates =  torch.linspace(0.1, 2.0, len(neighbors))
			decayed_intervals = [torch.exp(-decay_rate * interval) for decay_rate in decay_rates]
			neighbors = torch.from_numpy(neighbors).long().to(self.device)
			bs = neighbors.shape[0]
			neighbor_emb = self.memory.emb[neighbors.flatten()].view(bs, self.n_update_neighbors, self.emb_dim)
			# for interval in decayed_intervals:
			damp = 1 / (1 + self.gamma * interval)
			# neighbor_emb [bs][n_update_neighbors][emb_dim]
			damp = np.repeat(np.expand_dims(damp, axis=2), self.emb_dim, axis=2)
			neighbor_emb = neighbor_emb * torch.from_numpy(np.float32(damp)).to(self.device)

			# =====================================================================================
			damp1 = 1 / (1 + self.gamma * interval)
			damp1 = torch.tensor(damp1)
			# (16,20,64)
			changed_emb = model(h,damp1,neighbor_emb)
			# =====================================================================================

			x = torch.cat((neighbor_emb, changed_emb), dim=2)
			x.detach_()
			x = torch.matmul(x, self.W_1)
			x = x.relu()
			x = torch.matmul(x, self.W_2)
			probs = x.sigmoid()
			# (320,64)
			changed_emb = torch.matmul(torch.cat((self.memory.emb[neighbors.flatten()],
												changed_emb.flatten().view(-1, self.emb_dim)), dim=1), self.W_un).tanh()
				# changed_emb_lt.append(changed_emb)
			# changed_embs = [sum(x) for x in zip(*changed_emb_lt)]
			# changed_emb = sum(changed_emb_lt) / len(changed_emb_lt)
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
					self.memory.emb[neighbors.flatten()] = mask * changed_emb + (1 - mask) * self.memory.emb[
						neighbors.flatten()]
				else:
					# w/o-RNS arm: no gating, every candidate neighbour is updated.
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

		# compute loss
		pos_score, neg_score = self.compute_score(src_idxs, dst_idxs, neg_idxs)
		return pos_score, neg_score, policy_loss, reward

	def get_reward(self, idxs, neighbor_emb):
		reg_strength = 0.01
		central_emb = self.memory.emb[idxs].repeat(1, self.n_update_neighbors).view(-1, self.emb_dim)
		central_emb_norm = F.normalize(central_emb, p=2, dim=1).detach()
		neighbor_emb_norm = F.normalize(neighbor_emb, p=2, dim=1)
		cos_sim = torch.matmul(central_emb_norm, neighbor_emb_norm.t())
		# ======================================================================
		# 计算正则项，这里使用的是中心嵌入的 L2 范数
		reg_term = torch.norm(central_emb, p=2, dim=1).mean()

		# 奖励是余弦相似度的平均值减去正则项的缩放版本
		# self.reg_strength 是正则项的权重，控制正则项的强度
		reward = cos_sim.mean() - reg_strength * reg_term
		# ======================================================================
		return reward

	def compute_sim(self, src_idxs):
		src_norm = F.normalize(self.memory.emb[src_idxs], p=2, dim=1)
		emb_norm = F.normalize(self.memory.emb, p=2, dim=1)
		cos_sim = torch.matmul(src_norm, emb_norm.t())
		sorted_cos_sim, idx = cos_sim.sort(descending=True)
		return sorted_cos_sim, idx

	def compute_score(self, src_idxs, dst_idxs, neg_idxs):
		pos_score = torch.sum(self.memory.emb[src_idxs] * self.memory.emb[dst_idxs], dim=1)
		neg_score = torch.sum(self.memory.emb[src_idxs] * self.memory.emb[neg_idxs], dim=1)
		return pos_score.sigmoid(), neg_score.sigmoid()

	def reset_graph(self):
		# Restore the snapshot taken at construction rather than redrawing, so that
		# every epoch starts from the same memory state and the global RNG is left
		# untouched.
		self.memory.reset()

	def set_neighbor_finder(self, neighbor_finder):
		self.neighbor_finder = neighbor_finder

	def detach_memory(self):
		self.memory.detach_memory()

	def back_up_memory(self):
		return self.memory.emb.clone()

	def restore_memory(self, back_up):
		# requires_grad must be forced off: nn.Parameter defaults to True, and a
		# grad-requiring leaf would make the in-place write in forward() raise
		# "a leaf Variable that requires grad is being used in an in-place operation".
		self.memory.emb = nn.Parameter(back_up.detach().clone(), requires_grad=False)
