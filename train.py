from config import *
from dataload import *
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from munch import Munch
from sklearn.metrics import mean_squared_error
from fire import Fire

import os
import numpy as np

class STSModule(nn.Module):
	def __init__(self, model):
		super(LSTMSiamese, self).__init__()
		self.model = model
	def forward(self, x1, x2):
		x1 = self.model(x1)
		x2 = self.model(x2)
		x = mean_squared_error(x1, x2)
		return x
	
def main():
	random.seed(EXPCONF.seed)
	np.random.seed(EXPCONF.seed)
	torch.maual_seed(EXPCONF.seed)
	torch.cuda.manual_seed_all(EXPCONF.seed)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	dataloader, ds = get_loader(EXPCONF)

	assert len(dataloader)>0, f"dataloader is empty!"
	
	model = nn.LSTM(EXPCONF.input_size, EXPCONF.hidden_size, EXPCONF.num_layers).to(device)

	for ep in tqdm(range(1, EXPCONF.numep+1), desc="epoch progress")
		model.train()
		for i, (b, datasetids) in enumerate(tqdm(trainloader, desc="iterations progress"),1):
	
	

