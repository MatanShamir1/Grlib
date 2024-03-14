import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from types import MethodType
import numpy as np


def accuracy_per_batch(model, data_set):
    model.eval()
    correct = total = 0.0
    sum_loss = 0.0
    with torch.no_grad():
        for (first_traces, second_traces, is_same_goals) in data_set:
            y_pred = model(first_traces, second_traces)
            loss = F.binary_cross_entropy(y_pred, is_same_goals)
            sum_loss += loss.item()
            y_pred = (y_pred >= 0.5)
            correct += torch.sum(y_pred == is_same_goals)
    return correct / total, sum_loss / 32



class LstmObservations(nn.Module):
	def __init__(self, hidden_dim, batchsize,max_episode_length=20):
		super(LstmObservations,self).__init__()
		self.batch_size = batchsize
		self.lstm = nn.LSTM(hidden_size=hidden_dim, batch_first=True)

	def forward(self, trace1, trace2):
		out1,_ = self.lstm(trace1,None)
		out2,_ = self.lstm(trace2,None)
		manhattan_dis = torch.exp(-torch.sum(torch.abs(out1[:,-1,:]-out2[:,-1,:]),dim=1,keepdim=True))
		return manhattan_dis

def train_metric_model(model, train_loader, dev_loader, index_to_label, nepochs):
	devAccuracy = []
	optimizer = torch.optim.Adam(model.parameters(),weight_decay=1.25)
	for epoch in range(nepochs):
		model.train()
		for (first_traces, second_traces, is_same_goals) in train_loader:
			model.zero_grad()
			y_pred = model(first_traces, second_traces)
			loss = F.binary_cross_entropy(y_pred, is_same_goals)
			loss.backward()
			optimizer.step()
   
		dev_accuracy, dev_loss = accuracy_per_batch(model, dev_loader)
		devAccuracy.append(dev_accuracy)

		print("epoch - {}/{}...".format(epoch + 1, nepochs),
				"dev loss - {:.6f}...".format(dev_loss),
				"dev accuracy - {:.6f}".format(dev_accuracy))
  