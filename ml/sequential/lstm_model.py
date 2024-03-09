import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from types import MethodType


def get_accuracy(net,q1,q2,labels,batchsize):
	correct = 0
	total = 0

	numbatches = int(len(q1)/batchsize)

	for i in range(numbatches):
		x1 = q1[i*batchsize:(i+1)*batchsize,:]
		x2 = q2[i*batchsize:(i+1)*batchsize,:]

		y = labels[i*batchsize:(i+1)*batchsize]

		preds = net(x1,x2)
		preds = preds.view(preds.size(0))
		ypred = (preds>=0.5).double()

		y = y.double()

		correct+=torch.sum(y==ypred).item()

		total+=x1.size(0)

	return ((correct/total)*100)



class LstmObservations(nn.Module):
	def __init__(self, hidden_dim, batchsize,max_episode_length=20):
		super(LstmObservations,self).__init__()
		self.batch_size = batchsize
		self.lstm = nn.LSTM(hidden_size=hidden_dim, batch_first=True)

	def forward(self,x1,x2):
		q1_emb = self.embedding_layer(x1)
		q2_emb = self.embedding_layer(x2)
		q1_out,_ = self.lstm(q1_emb,None)
		q2_out,_ = self.lstm(q2_emb,None)

		manhattan_dis = torch.exp(-torch.sum(torch.abs(q1_out[:,-1,:]-q2_out[:,-1,:]),dim=1,keepdim=True))

		return manhattan_dis

def train_metric_model(model, train_loader, dev_loader, index_to_label, nepochs):
	devAccuracy = []
	optimizer = torch.optim.Adam(model.parameters(),weight_decay=1.25)
	for epoch in range(nepochs):
		model.train()
		counter = 0
		sum_loss = 0.0
		for batch_idx, (x, x_chars, pref_x, suf_x, y, lengths) in enumerate(train_loader):
			counter += x.shape[0]
			model.zero_grad()
			outputs, y = prepare_to_forward(model, x, x_chars, pref_x, suf_x, y, lengths)
			loss = F.nll_loss(outputs, y.view(-1) - 1, ignore_index=-1)
			sum_loss += loss.item()
			loss.backward()
			optimizer.step()
			if counter % 500 == 0:
				dev_accuracy, dev_loss = accuracy_per_batch(model, dev_loader, index_to_label)
				devAccuracy.append(dev_accuracy)
				if dev_accuracy > best_accuracy:
					best_accuracy = dev_accuracy
					torch.save(model.state_dict(), model_path)

				print("epoch - {}/{}...".format(epoch + 1, epochs),
						"step - {}...".format(counter),
						"dev loss - {:.6f}...".format(dev_loss),
						"dev accuracy - {:.6f}".format(dev_accuracy))
				model.train()
	numbatches = int(len(train_ind1)/batchsize)

	for epoch in range(numepochs):
		for i in range(numbatches):
			train_q1 = train_ind1[i*batchsize:(i+1)*batchsize,:]
			train_q2 = train_ind2[i*batchsize:(i+1)*batchsize,:]

			ytrain = train_labels[i*batchsize:(i+1)*batchsize]

			ytrain = ytrain.view(-1,1).float()

			ypred = model(train_q1,train_q2)

			ypred = ypred.float()

			loss = F.binary_cross_entropy(ypred,ytrain)

			loss.backward()

			optimizer.step()

		train_acc = get_accuracy(model,train_ind1,train_ind2,train_labels,batchsize)

		print("Train Loss {} and Train Accuracy {}".format(loss,train_acc))

		valid_acc = get_accuracy(model,valid_ind1,valid_ind2,valid_labels,batchsize)

		print("Validation Loss {} and Validation Accuracy {}".format(loss,valid_acc))

	test_acc = get_accuracy(model,test_ind1,test_ind2,test_labels,batchsize)

	print("Test Loss {} and Test Accuracy {}".format(loss,test_acc))
 
def accuracy_per_batch(model, data_set, index_to_label):
	model.eval()
	good = total = 0.0
	sum_loss = 0.0
	with torch.no_grad():
		for batch_idx, (x, x_chars, pref_x, suf_x, y, lengths) in enumerate(data_set):
			outputs, y = prepare_to_forward(model, x, x_chars, pref_x, suf_x, y, lengths)
			outputs = outputs.detach().cpu()
			y = y.cpu()
			predictions = np.argmax(outputs.data.numpy(), axis=1)
			loss = F.nll_loss(outputs, y.view(-1) - 1, ignore_index=-1)
			sum_loss += loss.item()
			total += sum(lengths)
			for y_hat, label in np.nditer([predictions + 1, y.view(-1).numpy()]):
				if label != 0:
					if y_hat == label:

						# Don't count the cases in which both prediction and label are 'O' in NER.
						if (not is_pos) and index_to_label[int(y_hat)] == 'O':
							total -= 1
						else:
							good += 1

	return good / total, sum_loss / dev_batch