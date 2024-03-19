import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from types import MethodType
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
	def __init__(self, hidden_dim=32, batchsize=32):
		super(LstmObservations,self).__init__()
		self.batch_size = batchsize
		self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_dim, batch_first=True)

	def forward(self, traces1, traces2, lengths1, lengths2):
		out1,_ = self.lstm(pack_padded_sequence(traces1, lengths1, batch_first=True, enforce_sorted=False), None)
		out2,_ = self.lstm(pack_padded_sequence(traces2, lengths2, batch_first=True, enforce_sorted=False), None)
		out1, _ = pad_packed_sequence(out1, batch_first=True, total_length=max(lengths1))
		out2, _ = pad_packed_sequence(out2, batch_first=True, total_length=max(lengths2))
		manhattan_dis = torch.exp(-torch.sum(torch.abs(out1[:,-1,:]-out2[:,-1,:]),dim=1,keepdim=True))
		return manhattan_dis

def train_metric_model(model, train_loader, dev_loader, nepochs=10):
	devAccuracy = []
	optimizer = torch.optim.Adam(model.parameters(),weight_decay=1.25)
	for epoch in range(nepochs):
		model.train()
		for (first_traces, second_traces, is_same_goals, lengths1, lengths2) in train_loader:
			model.zero_grad()
			y_pred = model(first_traces, second_traces, lengths1, lengths2)
			loss = F.binary_cross_entropy(y_pred, is_same_goals)
			loss.backward()
			optimizer.step()
   
		dev_accuracy, dev_loss = accuracy_per_batch(model, dev_loader)
		devAccuracy.append(dev_accuracy)

		print("epoch - {}/{}...".format(epoch + 1, nepochs),
				"dev loss - {:.6f}...".format(dev_loss),
				"dev accuracy - {:.6f}".format(dev_accuracy))
  