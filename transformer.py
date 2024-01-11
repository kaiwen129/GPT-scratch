import torch
import torch.nn as nn
import torch.nn.functional as F 

from modules import DecoderBlock

class GenTRF(nn.Module):
	def __init__(self, k, heads, seq_length, num_tokens, depth=8):
		super().__init__()

		self.num_tokens = num_tokens
		self.seq_length = seq_length
		self.token_embedding = nn.Embedding(num_tokens, k)
		self.pos_embedding = nn.Embedding(seq_length, k)

		blocks = []
		for i in range(depth):
			blocks.append(DecoderBlock(k=k, heads=heads, mask=True))
		self.stack = nn.Sequential(*blocks)

		self.toprobs = nn.Linear(k, num_tokens)
	
	def forward(self, x):
		tokens = self.token_embedding(x)
		b, t, k = tokens.size()

		positions = torch.arange(self.seq_length)
		positions = self.pos_embedding(positions)
		positions = positions[None, :, :].expand(b, t, k)

		x = tokens + positions
		x = self.stack(x)

		x = self.toprobs(x.reshape(b*t, k)).reshape(b, t, self.num_tokens)

		return F.log_softmax(x, dim=2)