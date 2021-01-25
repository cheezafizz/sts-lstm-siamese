import os
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from munch import Munch
from collections import defaultdict

from config import *
from utils import *

class STSDataset(Dataset):

	def __init__(self, mode):
		self.dataframe = pd.read_csv("../data/train.csv")
		tsz = NearestMultiple(EXPCONF.tsz, EXPCONF.bsz)
		dsz = NearestMultiple(EXPCONF.dsz, EXPCONF.bsz)
		self.dataframe = self.dataframe.iloc[:tsz, :] if mode else self.dataframe.iloc[tsz:dsz, :] 
	
	def __len__(self):
		return len(self.dataframe)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx= idx.tolist()
		sentence1 = self.dataframe.iloc[idx,1].split()
		sentence1 = [int(x) for x in sentence1]
		sentence2 = self.dataframe.iloc[idx,2].split()
		sentence2 = [int(x) for x in sentence2]

		return {'sentence1': sentence1, 'sentence2': sentence2, 'label': self.dataframe.iloc[idx,3]}
	
	def collate(self, data):
		batch = defaultdict(list)
		device = torch.cuda if torch.cuda.is_available() else torch
		for d in data:
			batch['sentence1'].append(device.LongTensor(d['sentence1']))
			batch['sentence2'].append(device.LongTensor(d['sentence2']))
			batch['label'].append(d['label'])
			batch['len1'].append(len(d['sentence1']))
			batch['len2'].append(len(d['sentence2']))
		batch['sentence1'] = pad_sequence(batch['sentence1'], padding_value=1)
		batch['sentence2'] = pad_sequence(batch['sentence2'], padding_value=1)
		batch['label'] = device.LongTensor(batch['label'])

		return Munch(batch)


		
def get_loader(expconf, mode):
	print("preparing dataloader")
	ds = STSDataset(mode)
	loader = DataLoader(dataset = ds, batch_size = expconf.bsz, shuffle = True, num_workers= expconf.numworkers, collate_fn = ds.collate)
	return list(loader), ds









			


		


