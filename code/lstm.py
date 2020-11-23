import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTM(nn.Module):
	"""docstring for LSTM"""
	def __init__(self, in_dim=25002,out_dim=1,embedding_dim=100, hidden_dim=256, pretrained_emb=None, dropout=0.25):
		super(LSTM, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.dropout = nn.Dropout(dropout)


		self.embedding = nn.Embedding(in_dim, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=3, bidirectional=True, dropout=dropout)
		self.projection = nn.Linear(2*hidden_dim, out_dim)

		if pretrained_emb is not None:
			self.embedding.weight.data.copy_(pretrained_emb)

	def forward(self, x):
		x = self.embedding(x)
		x = self.dropout(x)
		y, (h, c) = self.lstm(x)
		h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
		hidden = self.dropout(h)
		res = self.projection(hidden.squeeze(0))
		return res




		