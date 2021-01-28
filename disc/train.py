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

	train_loader, train_ds = get_loader(EXPCONF, EXPCONF.train_mode)
	model = nn.LSTM(EXPCONF.input_size, EXPCONF.hidden_size, EXPCONF.num_layers).to(device)
	model = STSModule(model)
	model = model.to(device)

	grouped_params = [{
		"params": [p for n, p in model.named_parameters()],
		"weight_decay": EXPCONF.weight_decay,
		}]
	optimizer = AdamW(grouped_params, lr = EXPCONF.lr)
	wandb.watch(model)

	#train
	for ep in tqdm(range(1, EXPCONF.numep+1), desc="epoch progress"):
		model.train()
		for i, b in enumerate(tqdm(train_loader, desc="iterations progress"),1):
			# batch learning
			optimizer.zero_grad()
			output1, output2 = model(b.sentence1, b.sentence2, b.len1, b.len2)
			loss = model.loss_fn(output1, output2, b.label)
			loss.mean().backward()
			optimizer.step()

	#save model
	savemodel(EXPCONF, model)
	
		
	#eval
	eval_loader, eval_ds = get_loader(EXPCONF, EXPCONF.eval_mode)
	model.eval()
	correct = 0
	cnt = 0
	for i, b in enumerate(tqdm(eval_loader, desc="eval progress"),1):
		output1, output2 = model(b.sentence1, b.sentence2, b.len1, b.len2)
		loss = model.loss_fn(output1, output2, b.label)
		for l in loss:
			if l<0.5:
				correct += 1
			cnt += 1
	print("Accuracy : ", correct/cnt*100, "%") 


if __name__ == '__main__':
		main()
