import torch
import numpy as np
from torch import nn

class Memory(nn.Module):

    def __init__(self, n_nodes, emb_dim, device='cpu'):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.emb_dim = emb_dim
        self.device = device
        self.__init_memory__()

    def __init_memory__(self):
        emb = torch.empty((self.n_nodes, self.emb_dim), device=self.device)
        nn.init.xavier_normal_(emb)
        self.emb = nn.Parameter(emb, requires_grad=False)
        self._init_snapshot = emb.detach().clone()

    def reset(self):
        with torch.no_grad():
            self.emb.copy_(self._init_snapshot)

    def detach_memory(self):
        self.emb.detach_()
