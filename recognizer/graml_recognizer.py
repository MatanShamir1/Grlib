from abc import ABC
import math
import os
import random
from dataset.gr_dataset import GRDataset, generate_datasets
from ml import utils
from ml.base import RLAgent
from metrics import metrics
from typing import List, Tuple, Type
from types import MethodType
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from ml.planner.mcts import mcts_model

from ml.sequential.lstm_model import LstmObservations, train_metric_model, train_metric_model_cont
from ml.utils.storage import get_model_dir, problem_list_to_str_tuple

### IMPLEMENT MORE SELECTION METHODS, MAKE SURE action_probs IS AS IT SEEMS: list of action-probability 'es ###

def collate_fn(batch):
	first_traces, second_traces, is_same_goals = zip(*batch)
	# torch.stack takes tensor tuples (fixed size) and stacks them up in a matrix
	first_traces_padded = pad_sequence([torch.stack(sequence) for sequence in first_traces], batch_first=True)
	second_traces_padded = pad_sequence([torch.stack(sequence) for sequence in second_traces], batch_first=True)
	first_traces_lengths = [len(trace) for trace in first_traces]
	second_traces_lengths = [len(trace) for trace in second_traces]
	return first_traces_padded.to(utils.device), second_traces_padded.to(utils.device), torch.stack(is_same_goals).to(utils.device), first_traces_lengths, second_traces_lengths

def collate_fn_cont(batch):
	first_traces, second_traces, is_same_goals = zip(*batch)
	# pad sequence takes a batch of sequences and pads all of them to be as long as the longest sequence. with zeros.
	# a pile of torches of different sizes, each one a 'list' of images\texts is padded, which means there are "0" images in the padded traces.
	first_traces_images_padded = pad_sequence([torch.stack([step.image for step in sequence]) for sequence in first_traces], batch_first=True)
	first_traces_texts_padded = pad_sequence([torch.stack([step.text for step in sequence]) for sequence in first_traces], batch_first=True)
	second_traces_images_padded = pad_sequence([torch.stack([step.image for step in sequence]) for sequence in second_traces], batch_first=True)
	second_traces_texts_padded = pad_sequence([torch.stack([step.text for step in sequence]) for sequence in second_traces], batch_first=True)
	first_traces_lengths = [len(trace) for trace in first_traces]
	second_traces_lengths = [len(trace) for trace in second_traces]
	return first_traces_images_padded.to(utils.device), first_traces_texts_padded.to(utils.device), second_traces_images_padded.to(utils.device), second_traces_texts_padded.to(utils.device), \
			torch.stack(is_same_goals).to(utils.device), first_traces_lengths, second_traces_lengths

def goal_to_minigrid_str(tuply):
	tuply = tuply[1:-1] # remove the braces
	#print(tuply)
	nums = tuply.split(',')
	#print(nums)
	return f'MiniGrid-SimpleCrossingS13N4-DynamicGoal-{nums[0]}x{nums[1]}-v0'

def load_weights(loaded_model : LstmObservations, path):
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	loaded_model.load_state_dict(torch.load(path, map_location=utils.device))
	loaded_model.to(utils.device)  # Ensure model is on the right device
	return loaded_model

def save_weights(model : LstmObservations, path):
	directory = os.path.dirname(path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	torch.save(model.state_dict(), path)

class GramlRecognizer(ABC):
	def __init__(self, method: Type[RLAgent], env_name: str, problems: List[str], grid_size, is_continuous = False):
		self.problems = problems
		self.env_name = env_name
		self._observability_percentages = [0.1, 0.3, 0.5, 0.7, 1.0]
		self.rl_agents_method = method
		self.agents: List[RLAgent] = []
		self.is_continuous = is_continuous
		if is_continuous: self.train_func = train_metric_model_cont; self.collate_func = collate_fn_cont
		else: self.train_func = train_metric_model; self.collate_func = collate_fn

	# TODO create a cache for the learning phase and load the weights if the problem was alread learnt.
	def domain_learning_phase(self):
		# start by training each rl agent on the base goal set
		for problem_name in self.problems:
			agent = self.rl_agents_method(env_name=self.env_name, problem_name=problem_name)
			agent.learn()
			self.agents.append(agent)
		self.obs_space, self.preprocess_obss = utils.get_obss_preprocessor(self.agents[0].env.observation_space)
		# train the network so it will find a metric for the observations of the base agents such that traces of agents to different goals are far from one another		
		self.model_directory = get_model_dir(env_name=self.env_name, model_name=problem_list_to_str_tuple(self.problems), class_name=self.__class__.__name__)
		if self.is_continuous : last_path = r"lstm_cnn_model_cont.pth"
		else: last_path = r"lstm_cnn_model.pth"
		self.model_file_path = os.path.join(self.model_directory, last_path)
		self.model = LstmObservations(obs_space=self.obs_space, action_space=self.agents[0].env.action_space, is_continuous=self.is_continuous)
		self.model.to(utils.device)

		if os.path.exists(self.model_file_path):
			print(f"Loading pre-existing lstm model in {self.model_file_path}")
			load_weights(loaded_model=self.model, path=self.model_file_path)
		else:
			train_samples, dev_samples = generate_datasets(100000, self.agents, metrics.stochastic_amplified_selection, problem_list_to_str_tuple(self.problems), self.env_name, self.preprocess_obss, self.is_continuous)
			train_dataset = GRDataset(len(train_samples), train_samples)
			dev_dataset = GRDataset(len(dev_samples), dev_samples)
			self.train_func(self.model,	train_loader=DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=self.collate_func),
							dev_loader=DataLoader(dev_dataset, batch_size=64, shuffle=False, collate_fn=self.collate_func))
			save_weights(model=self.model, path=self.model_file_path)

	def goals_adaptation_phase(self, new_goals):
		self.current_goals = new_goals
		# start by training each rl agent on the base goal set
		problems_goals = [(goal_to_minigrid_str(tuply),tuply) for tuply in new_goals]
		self.embeddings_dict = {}
		# will change, for now let an optimal agent give us the trace
		for (problem_name, goal) in problems_goals:
			# agent = self.rl_agents_method(env_name=self.env_name, problem_name=problem_name)
			# agent.learn()
			# obs = agent.generate_partial_observation(action_selection_method=metrics.greedy_selection, percentage=random.choice([0.5, 0.7, 1]))
			obs = mcts_model.plan(self.env_name, problem_name)
			if self.is_continuous: embedding = self.model.embed_sequence_cont(obs, self.preprocess_obss)
			else: embedding = self.model.embed_sequence(obs)
			self.embeddings_dict[goal] = embedding

	def inference_phase(self, sequence):
		if self.is_continuous: new_embedding = self.model.embed_sequence_cont(sequence, self.preprocess_obss)
		else: new_embedding = self.model.embed_sequence(sequence)
		closest_goal, greatest_similarity = None, 0
		for (goal, embedding) in self.embeddings_dict.items():
			curr_similarity = torch.exp(-torch.sum(torch.abs(embedding-new_embedding)))
			if curr_similarity > greatest_similarity:
				print(f'new closest goal is: {goal}')
				closest_goal = goal
				greatest_similarity = curr_similarity
		return closest_goal