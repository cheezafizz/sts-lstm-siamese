from config import *
from dataload_disc import *
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
from wandb import wandb
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
	'''
	def loss_fn(self, x1, x2, label):
		cos = CosineDistance(x1, x2, -1)

		for i in range(EXPCONF.bsz):
			cos[i] = cos[i] if label[i] else 1-cos[i]

		return cos
	'''

class Discriminator(nn.Module):
	def __init__(self, device):
		super(Discriminator,self).__init__()
		self.device = device

	def forward(self, x1, x2):
		x3 = torch.cat((x1, x2), -1)
		size = list(x3.shape)
		
		lin1 = nn.Linear(size[-1], 16).to(self.device)
		lin2 = nn.Linear(16, 2).to(self.device)
		soft = nn.Softmax(dim=-1)

		x3= lin1(x3)
		x3= lin2(x3)
		x3= torch.squeeze(x3)
		x3= soft(x3)
		return x3
	
	def loss_fn(self, x1, x2):
		loss = nn.BCELoss()
		return loss(x1, x2)
	
def main():
	# wandb.init()
	random.seed(EXPCONF.seed)
	np.random.seed(EXPCONF.seed)
	torch.manual_seed(EXPCONF.seed)
	torch.cuda.manual_seed_all(EXPCONF.seed)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_loader, train_ds = get_loader(EXPCONF, EXPCONF.train_mode)
	model = nn.LSTM(EXPCONF.input_size, EXPCONF.hidden_size, EXPCONF.num_layers).to(device)
	model = STSModule(model)
	model = model.to(device)
	disc = Discriminator(device).to(device)

	grouped_params = [{
		"params": [p for n, p in model.named_parameters()],
		"weight_decay": EXPCONF.weight_decay,
		}]
	grouped_params_disc = [{
		"params": [p for n, p in disc.named_parameters()],
		"weight_decay": EXPCONF.weight_decay,
		}]
	optimizer = AdamW(grouped_params, lr = EXPCONF.lr)
	optimizer_disc = AdamW(grouped_params_disc, lr = EXPCONF.lr)
	# wandb.watch(model)

	#train discriminator
	dev_loader, dev_ds = get_loader(EXPCONF, EXPCONF.dev_mode)
	for ep in tqdm(range(1, EXPCONF.numep+1), desc="epoch progress DISC"):
		disc.train()
		for i, b in enumerate(tqdm(dev_loader, desc="iterations progress DISC"),1):
			optimizer.zero_grad()
			optimizer_disc.zero_grad()
			output1, output2 = model(b.sentence1, b.sentence2, b.len1, b.len2)
			output3 = disc(output1, output2)
			label = torch.stack(b.label, dim=0)
			loss = disc.loss_fn(output3, label)
			loss.mean().backward()
			optimizer_disc.step()

	#train
	for ep in tqdm(range(1, EXPCONF.numep+1), desc="epoch progress MODEL"):
		model.train()
		for i, b in enumerate(tqdm(train_loader, desc="iterations progress MODEL"),1):
			# batch learning
			optimizer.zero_grad()
			optimizer_disc.zero_grad()
			output1, output2 = model(b.sentence1, b.sentence2, b.len1, b.len2)
			output3 = disc(output1, output2)
			label = torch.stack(b.label, dim=0)
			loss = disc.loss_fn(output3, label)
			loss.mean().backward()
			optimizer.step()
	
	#eval
	eval_loader, eval_ds = get_loader(EXPCONF, EXPCONF.eval_mode)
	model.eval()
	disc.eval()
	correct = 0
	cnt = 0
	for i, b in enumerate(tqdm(eval_loader, desc="eval progress"),1):
		output1, output2 = model(b.sentence1, b.sentence2, b.len1, b.len2)
		output3 = disc(output1, output2)
		for i, x in enumerate(output3):
			if x[0]<0.5 and b.label[i][0]<0.5:
				correct += 1
			elif x[1]<0.5 and b.label[i][1]<0.5:
				correct += 1
			cnt += 1
	print("Accuracy : ", correct/cnt*100, "%") 


if __name__ == '__main__':
		main()
