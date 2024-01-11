import torch
import torch.nn as nn
import torch.nn.functional as F 

class MultiHeadAttention(nn.Module):
	def __init__(self, k, heads=4, mask=False):
		super().__init__()

		self.heads = heads
		self.mask = mask
		self.k = k # embedding dim

		assert k % heads == 0
		
		self.Wk = nn.Linear(k, k)
		self.Wq = nn.Linear(k, k)
		self.Wv = nn.Linear(k, k)

		self.combine_heads = nn.Linear(k,k)

	def forward(self, x):
		b, t, k = x.size() # batch size, # of entries (e.g. words in a sentence), embedding dim

		keys = self.Wk(x)
		queries = self.Wq(x)
		values = self.Wv(x)

		s = k // self.heads # chunk size for lower-dim projections

		keys = keys.reshape((b, t, self.heads, s))
		queries = queries.reshape((b, t, self.heads, s))
		values = values.reshape((b, t, self.heads, s))

		keys = keys.transpose(1,2).reshape(b * self.heads, t, s) # combine # of batches and # of heads per batch into one dim
		queries = queries.transpose(1,2).reshape(b * self.heads, t, s) # now each "batch" is of dim t x s
		values = values.transpose(1,2).reshape(b * self.heads, t, s)

		### Attention Formula ###
		dot = torch.bmm(queries, keys.transpose(1,2)) # dot.size() = (b*h, t, t)
		
		if self.mask:
			indices = torch.triu_indices(t, t, offset=1)
			dot[:, indices[0], indices[1]] = float('-inf')

		dot /= (k**(0.5))
		dot = F.softmax(dot, dim=2)
		output = torch.bmm(dot, values)
		#########################

		output = output.reshape(b, self.heads, t, s) # separate heads into its own dim again
		output = output.transpose(1,2).reshape(b, t, k) # swap head dim w t, concatenate heads together

		output = self.combine_heads(output) # combine heads (t, k) --> (t, k)
		return output # output.size() = (b, t, k)

## Turn into decoder block

class DecoderBlock(nn.Module):
	def __init__(self, k, heads, mask, ff_size=4):
		super().__init__()

		self.MHAttention = MultiHeadAttention(k, heads=heads)
		self.norm1 = nn.LayerNorm(k)
		self.norm2 = nn.LayerNorm(k)

		self.mask = mask

		self.ff = nn.Sequential(
			nn.Linear(k, ff_size*k),
			nn.ReLU(),
			nn.Linear(ff_size*k, k)
		)
	
	def forward(self, x):
		output = self.MHAttention(x)
		output = self.norm1(output + x)

		ff_output = self.ff(output)
		output = self.norm2(output + ff_output)

		return output
		