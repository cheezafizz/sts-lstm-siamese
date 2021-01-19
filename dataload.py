import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import *

class STSDataset(Dataset):

	def __init__(self, root_dir):
		self.dataframe = pd.read_csv("../data/train.csv")
	
	def __len__(self):
		return len(self.dataframe)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx= idx.tolist()
		return {'sentence1': self.dataframe.iloc[idx, 1], 'sentence2': self.dataframe.iloc[idx,2], 'label': self.dataframe.iloc[idx,3]}
	

def get_loader(expconf):
	print("preparing dataloader")
	ds = STSDataset()
	loader = DataLoader(dataset = ds, batch_size = expconf.bsz, shuffle = True, num_workers= expconf.numworkers)
	return list(loader), ds









			


		


