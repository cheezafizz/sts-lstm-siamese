from config import *
from utils import *

import torch
import torch.nn as nn
from pathlib import Path


class Discriminator(nn.Module):
	def __init__(self, device):
		super(Discriminator,self).__init__()
		self.device = device

	def forward(self, x1, x2):
		x3 = torch.cat((x1, x2), -1)
		size = list(x3.shape)
		
		lin1 = nn.Linear(size[-1], 32).to(self.device)
		lin2 = nn.Linear(32, 32).to(self.device)
		lin3 = nn.Linear(32, 2).to(self.device)
		soft = nn.Softmax(dim=-1)

		x3= lin1(x3)
		x3= lin2(x3)
		x3= lin3(x3)
		x3= torch.squeeze(x3)
		x3= soft(x3)
		return x3
	
	def loss_fn(self, x1, x2):
		loss = nn.BCELoss()
		return loss(x1, x2)


class STSModule(nn.Module):
	def __init__(self, model):
		super(STSModule, self).__init__()
		self.model = model
		self.embedding = nn.Embedding(EXPCONF.vocab_size, EXPCONF.embedding_dim)

	def forward(self, x1, x2, len1, len2):
		x1, (h1, c1) = self.model(self.embedding(x1))
		x2, (h2, c2) = self.model(self.embedding(x2))
		
		# fetch real terminal hidden states
		list1 = [torch.unsqueeze(x1[len1[i]-1, i, :], 0) for i in range(EXPCONF.bsz)]
		list2 = [torch.unsqueeze(x2[len2[i]-1, i, :], 0) for i in range(EXPCONF.bsz)]

		# print(torch.cat((*list1,),0).shape, torch.cat((*list2,),0).shape)

		return torch.cat((*list1,), 0), torch.cat((*list2,), 0)

	def loss_fn(self, x1, x2, label):
		
		cos = CosineDistance(x1, x2, -1)

		for i in range(EXPCONF.bsz):
			cos[i] = cos[i] if label[i] else 1-cos[i]

		return cos

def savemodel(expconf, model, acc = 0):
	d_expconf = expconf.toDict()
	savepath = Path(expconf.savepath)
	saved = dict()
	saved = {
		'expconf' : d_expconf,
		'model' : model.state_dict(),
		'acc' : acc
	}
	torch.save(saved, savepath)

