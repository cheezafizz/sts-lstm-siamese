import torch
import torch.nn as nn

def CosineDistance(x1, x2, dim):
	cos = nn.CosineSimilarity(dim= dim)
	return (1-cos(x1, x2))/2

def NearestMultiple(x1, x2):
	return x1-x1%x2
