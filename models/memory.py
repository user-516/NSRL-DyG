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
		"""Draw a fresh initial memory from the ambient RNG.

		This deliberately does NOT seed the global RNG.  The run seed is owned by the
		caller, and re-seeding here would defeat it twice over: it would override
		whatever the caller set, and reset_graph() routes through this path, so the
		RNG would be reset to the same state at the start of every epoch.

		A snapshot of the initial state is kept so that reset_graph() can restore it
		deterministically without consuming randomness.
		"""
		emb = torch.empty((self.n_nodes, self.emb_dim), device=self.device)
		nn.init.xavier_normal_(emb)
		self.emb = nn.Parameter(emb, requires_grad=False)
		self._init_snapshot = emb.detach().clone()

	def reset(self):
		"""Restore the initial memory state.  Deterministic and RNG-free."""
		with torch.no_grad():
			self.emb.copy_(self._init_snapshot)

	def detach_memory(self):
		self.emb.detach_()
