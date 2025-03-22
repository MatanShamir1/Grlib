import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from types import MethodType
import numpy as np
from gr_libs.ml.utils import device
from torch.nn.utils.rnn import pack_padded_sequence


def accuracy_per_epoch(model, data_loader):
	model.eval()
	correct = total = 0.0
	sum_loss = 0.0
	with torch.no_grad():
		for (first_traces, second_traces, is_same_goals, first_traces_lengths, second_traces_lengths) in data_loader:
			y_pred = model.forward_tab(first_traces, second_traces, first_traces_lengths, second_traces_lengths)
			loss = F.binary_cross_entropy(y_pred, is_same_goals)
			sum_loss += loss.item()
			y_pred = (y_pred >= 0.5)
			correct += torch.sum(y_pred == is_same_goals)
			total += len(is_same_goals)
	return correct / total, sum_loss / 32

def accuracy_per_epoch_cont(model, data_loader):
	model.eval()
	correct = total = 0.0
	sum_loss = 0.0
	with torch.no_grad():
		for (first_traces_images, first_traces_texts, second_traces_images, second_traces_texts, is_same_goals, first_traces_lengths, second_traces_lengths) in data_loader:
			y_pred = model.forward_cont(first_traces_images, first_traces_texts, second_traces_images, second_traces_texts, first_traces_lengths, second_traces_lengths)
			loss = F.binary_cross_entropy(y_pred, is_same_goals)
			sum_loss += loss.item()
			y_pred = (y_pred >= 0.5)
			correct += torch.sum(y_pred == is_same_goals)
			total += len(is_same_goals)
	return correct / total, sum_loss / 32

# class CNNImageEmbeddor(nn.Module):
# 	def __init__(self, obs_space, action_space, use_text=False):
# 		super().__init__()
# 		self.use_text = use_text
# 		self.image_conv = nn.Sequential(
# 			nn.Conv2d(3, 4, kernel_size=(3, 3), padding=1),  # Reduced filters, added padding
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(4, 4, (3, 3), padding=1),  # Reduced filters, added padding
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),  # Added additional pooling to reduce size
#             nn.Conv2d(4, 8, (3, 3), padding=1),  # Reduced filters, added padding
#             nn.ReLU(),
#             nn.BatchNorm2d(8)
# 		)
# 		n = obs_space["image"][0]
# 		m = obs_space["image"][1]
# 		self.image_embedding_size = ((n - 4) // 4 - 3) * ((m - 4) // 4 - 3) * 8
# 		if self.use_text:
# 			self.word_embedding_size = 32
# 			self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
# 			self.text_embedding_size = 128
# 			self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

	def forward(self, images, texts):
	 	# images shape: batch_size X max_sequence_len X sample_size. same for text.
		# need to reshape image to num_channels X height X width, like nn.Conv expects it to be.
		x = images.transpose(2, 4).transpose(3, 4)
		orig_shape = x.shape
		# combine batch and sequence to 1 dimension so conv could handle it
		x = x.view(orig_shape[0]*orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4]) # x shape: batch_size * max_sequence_len X sample_size
		x = self.image_conv(x) # x shape:  batch_size * max_sequence_len X last_conv_size X 1 X 1
		# reshape x back to divide batches from sequences
		x = x.view(orig_shape[0], orig_shape[1], x.shape[1]) # x shape: batch_size X max_sequence_len X last_conv_size. last 2 dimensions (1,1) are collapsed to last conv.
		embedding = x
  
		if self.use_text:
			embed_text = self._get_embed_text(texts)
			embedding = torch.cat((embedding, embed_text), dim=1)
			
		return embedding

	def _get_embed_text(self, text):
		_, hidden = self.text_rnn(self.word_embedding(text))
		return hidden[-1]

class LstmObservations(nn.Module):
	
	def __init__(self, input_size, hidden_size): # TODO make sure the right cuda is used!
		super(LstmObservations,self).__init__()
		#self.embeddor = CNNImageEmbeddor(obs_space, action_space)
		# check if the traces are a bunch of images
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
		self.dropout = nn.Dropout(0.5)  # Added dropout layer
		# Initialize weights
		for name, param in self.lstm.named_parameters():
			if 'weight' in name:
				nn.init.xavier_uniform_(param)
			elif 'bias' in name:
				nn.init.zeros_(param)
		

	# tabular
	def forward_tab(self, traces1, traces2, lengths1, lengths2):
		out1, (ht1, ct1) = self.lstm(pack_padded_sequence(traces1, lengths1, batch_first=True, enforce_sorted=False), None) # traces1 & traces 2 shapes: batch_size X max sequence_length X embedding_size
		out2, (ht2, ct2) = self.lstm(pack_padded_sequence(traces2, lengths2, batch_first=True, enforce_sorted=False), None)
		# out1, _ = pad_packed_sequence(out1, batch_first=True, total_length=max(lengths1))
		# out2, _ = pad_packed_sequence(out2, batch_first=True, total_length=max(lengths2))
		manhattan_dis = torch.exp(-torch.sum(torch.abs(ht1[-1]-ht2[-1]),dim=1,keepdim=True))
		return manhattan_dis.squeeze()

	# continuous
	# def forward_cont(self, traces1_images, traces1_texts, traces2_images, traces2_texts, lengths1, lengths2):
	#  	# we also embed '0' images, but we take them out of the equation in the lstm (it knows to not treat them when batching)
	# 	traces1 = self.embeddor(traces1_images, traces1_texts)
	# 	traces2 = self.embeddor(traces2_images, traces2_texts) # traces1 & traces 2 shapes: batch_size X max_sequence_length X embedding_size
	# 	out1, (ht1, ct1) = self.lstm(pack_padded_sequence(traces1, lengths1, batch_first=True, enforce_sorted=False), None)
	# 	out2, (ht2, ct2) = self.lstm(pack_padded_sequence(traces2, lengths2, batch_first=True, enforce_sorted=False), None)
	# 	manhattan_dis = torch.exp(-torch.sum(torch.abs(ht1[-1]-ht2[-1]),dim=1,keepdim=True))
	# 	return manhattan_dis.squeeze()

	def embed_sequence(self, trace):
		trace = torch.stack([torch.tensor(observation, dtype=torch.float32) for observation in trace]).to(device)
		out, (ht, ct) = self.lstm(trace, None)
		return ht[-1]

	# def embed_sequence_cont(self, sequence, preprocess_obss):
	# 	sequence = [preprocess_obss([obs])[0] for ((obs, (_, _)), _) in sequence]
	# 	trace_images = torch.tensor(np.expand_dims(torch.stack([step.image for step in sequence]), axis=0)).to(device)
	# 	trace_texts = torch.tensor(np.expand_dims(torch.stack([step.text for step in sequence]), axis=0)).to(device)
	# 	embedded_trace = self.embeddor(trace_images, trace_texts)
	# 	out, (ht, ct) = self.lstm(embedded_trace)
	# 	return ht[-1]

def train_metric_model(model, train_loader, dev_loader, nepochs=5, patience = 2):
	devAccuracy = []
	best_dev_accuracy = 0.0
	no_improvement_count = 0
	optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=0.1)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
	for epoch in range(nepochs):
		sum_loss, denominator = 0.0, 0.0
		model.train()
		for (first_traces, second_traces, is_same_goals, first_traces_lengths, second_traces_lengths) in train_loader:
			model.zero_grad()
			y_pred = model.forward_tab(first_traces, second_traces, first_traces_lengths, second_traces_lengths)
			if len(is_same_goals) == 1: is_same_goals = torch.squeeze(is_same_goals) # for the case of batches in size 1...
			loss = F.binary_cross_entropy(y_pred, is_same_goals)
			sum_loss += loss.item()
			denominator += 1
			loss.backward()
			optimizer.step()
   
		dev_accuracy, dev_loss = accuracy_per_epoch(model, dev_loader)
		devAccuracy.append(dev_accuracy)
		if dev_accuracy > best_dev_accuracy:
			best_dev_accuracy = dev_accuracy
			no_improvement_count = 0
		else:
			no_improvement_count = 1

		print("epoch - {}/{}...".format(epoch + 1, nepochs),
				"train loss - {:.6f}...".format(sum_loss / denominator),
				"dev loss - {:.6f}...".format(dev_loss),
				"dev accuracy - {:.6f}".format(dev_accuracy))

		if no_improvement_count >= patience:
			print(f"Early stopping after {epoch + 1} epochs with no improvement.")
			break
  
def train_metric_model_cont(model, train_loader, dev_loader, nepochs=5):
	devAccuracy = []
	optimizer = torch.optim.Adadelta(model.parameters(),weight_decay=1.25)
	for epoch in range(nepochs):
		sum_loss, denominator = 0.0, 0.0
		model.train()
		for (first_traces_images, first_traces_texts, second_traces_images, second_traces_texts, is_same_goals, first_traces_lengths, second_traces_lengths) in train_loader:
			model.zero_grad()
			y_pred = model.forward_cont(first_traces_images, first_traces_texts, second_traces_images, second_traces_texts, first_traces_lengths, second_traces_lengths)
			loss = F.binary_cross_entropy(y_pred, is_same_goals)
			sum_loss += loss.item()
			denominator += 1
			loss.backward()
			optimizer.step()
   
		dev_accuracy, dev_loss = accuracy_per_epoch_cont(model, dev_loader)
		devAccuracy.append(dev_accuracy)

		print("epoch - {}/{}...".format(epoch + 1, nepochs),
				"train loss - {:.6f}...".format(sum_loss / denominator),
				"dev loss - {:.6f}...".format(dev_loss),
				"dev accuracy - {:.6f}".format(dev_accuracy))
  