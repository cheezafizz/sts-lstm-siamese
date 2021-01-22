from config import *
from dataload import *
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from munch import Munch
from fire import Fire

import os
import numpy as np
import random
from tqdm import tqdm

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

	
def main():
	random.seed(EXPCONF.seed)
	np.random.seed(EXPCONF.seed)
	torch.manual_seed(EXPCONF.seed)
	torch.cuda.manual_seed_all(EXPCONF.seed)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	dataloader, ds = get_loader(EXPCONF)
	model = nn.LSTM(EXPCONF.input_size, EXPCONF.hidden_size, EXPCONF.num_layers).to(device)
	model = STSModule(model)
	model = model.to(device)

	grouped_params = [{
		"params": [p for n, p in model.named_parameters()],
		"weight_decay": EXPCONF.weight_decay,
		}]
	optimizer = AdamW(grouped_params, lr = EXPCONF.lr)

	for ep in tqdm(range(1, EXPCONF.numep+1), desc="epoch progress"):
		model.train()
		for i, b in enumerate(tqdm(dataloader, desc="iterations progress"),1):
			# batch learning
			optimizer.zero_grad()
			output1, output2 = model(b.sentence1, b.sentence2, b.len1, b.len2)
			loss = model.loss_fn(output1, output2, b.label)
			loss.backward()
			optimizer.step()


if __name__ == '__main__':
		main()
