from config import *
from dataload import *
from utils import *
from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from munch import Munch
from fire import Fire

import os
import numpy as np
import random
from pathlib import Path
from wandb import wandb
from tqdm import tqdm


def main():
	wandb.init()
	random.seed(EXPCONF.seed)
	np.random.seed(EXPCONF.seed)
	torch.manual_seed(EXPCONF.seed)
	torch.cuda.manual_seed_all(EXPCONF.seed)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_loader, train_ds = get_loader(EXPCONF, EXPCONF.train_mode, disc = True)
	eval_loader, eval_ds = get_loader(EXPCONF, EXPCONF.eval_mode, disc = True)

	model = nn.LSTM(EXPCONF.input_size, EXPCONF.hidden_size, EXPCONF.num_layers).to(device)
	model = STSModule(model)
	model = model.to(device)
	disc = Discriminator(device).to(device)

	grouped_params = [{
		"params": [p for n, p in model.named_parameters()]+[p for n, p in disc.named_parameters()],
		"weight_decay": EXPCONF.weight_decay,
		}]
	optimizer = AdamW(grouped_params, lr = EXPCONF.lr)
	wandb.watch(model)

	#train
	for ep in tqdm(range(1, EXPCONF.numep+1), desc="epoch progress"):
		#train
		model.train()
		disc.train()
		for i, b in enumerate(tqdm(train_loader, desc="iterations progress"),1):
			# batch learning
			optimizer.zero_grad()
			output1, output2 = model(b.sentence1, b.sentence2, b.len1, b.len2)
			output3 = disc(output1, output2)
			label = torch.stack(b.label, dim=0)
			loss = disc.loss_fn(output3, label)
			loss.mean().backward()
			optimizer.step()
		#validate
		model.eval()
		disc.eval()
		cnt = 0
		correct = 0
		for i, b in enumerate(tqdm(eval_loader, desc="validation progress"),1):
			output1, output2 = model(b.sentence1, b.sentence2, b.len1, b.len2)
			output3 = disc(output1, output2)
			for (x,y) in zip(output3,b.label):
				if x[0]<x[1] and y[0]<y[1]:
					correct += 1
				elif x[0]>x[1] and y[0]>y[1]:
					correct += 1
				cnt += 1
		acc = correct/cnt*100
		print("validation accuracy : ", acc, "%")
		wandb.log(
			{
				'accuracy': acc,
			}
		)	
	#eval
	model.eval()
	disc.eval()
	cnt = 0
	correct = 0
	for i, b in enumerate(tqdm(eval_loader, desc="evaluation progress"),1):
		output1, output2 = model(b.sentence1, b.sentence2, b.len1, b.len2)
		output3 = disc(output1, output2)
		for (x,y) in zip(output3,b.label):
			if x[0]<x[1] and y[0]<y[1]:
				correct += 1
			elif x[0]>x[1] and y[0]>y[1]:
				correct += 1
			cnt += 1
	acc = correct/cnt*100
	print("Evaluation accuracy : ", acc, "%")
	wandb.log(
		{
			'final_accuracy': acc,
		}
	)


if __name__ == '__main__':
		main()
