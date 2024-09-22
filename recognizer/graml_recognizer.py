from abc import ABC
import math
import os
import random

from stable_baselines3 import SAC
from dataset.graml.gr_dataset import GRDataset, generate_datasets
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
import dill

from ml.sequential.lstm_model import LstmObservations, train_metric_model, train_metric_model_cont
from ml.utils.format import minigrid_str_to_goal, random_subset_with_order
from ml.utils.storage import get_model_dir, get_embeddings_result_path

MCTS_BASED, AGENT_BASED, GC_AGENT_BASED = 0, 1, 2

### IMPLEMENT MORE SELECTION METHODS, MAKE SURE action_probs IS AS IT SEEMS: list of action-probability 'es ###

def collate_fn(batch):
	first_traces, second_traces, is_same_goals = zip(*batch)
	# torch.stack takes tensor tuples (fixed size) and stacks them up in a matrix
	first_traces_padded = pad_sequence([torch.stack(sequence) for sequence in first_traces], batch_first=True)
	second_traces_padded = pad_sequence([torch.stack(sequence) for sequence in second_traces], batch_first=True)
	first_traces_lengths = [len(trace) for trace in first_traces]
	second_traces_lengths = [len(trace) for trace in second_traces]
	return first_traces_padded.to(utils.device), second_traces_padded.to(utils.device), torch.stack(is_same_goals).to(utils.device), first_traces_lengths, second_traces_lengths

# def collate_fn_cont(batch):
# 	first_traces, second_traces, is_same_goals = zip(*batch)
# 	# pad sequence takes a batch of sequences and pads all of them to be as long as the longest sequence. with zeros.
# 	# a pile of torches of different sizes, each one a 'list' of images\texts is padded, which means there are "0" images in the padded traces.
# 	first_traces_images_padded = pad_sequence([torch.stack([step.image for step in sequence]) for sequence in first_traces], batch_first=True)
# 	first_traces_texts_padded = pad_sequence([torch.stack([step.text for step in sequence]) for sequence in first_traces], batch_first=True)
# 	second_traces_images_padded = pad_sequence([torch.stack([step.image for step in sequence]) for sequence in second_traces], batch_first=True)
# 	second_traces_texts_padded = pad_sequence([torch.stack([step.text for step in sequence]) for sequence in second_traces], batch_first=True)
# 	first_traces_lengths = [len(trace) for trace in first_traces]
# 	second_traces_lengths = [len(trace) for trace in second_traces]
# 	return first_traces_images_padded.to(utils.device), first_traces_texts_padded.to(utils.device), second_traces_images_padded.to(utils.device), second_traces_texts_padded.to(utils.device), \
# 			torch.stack(is_same_goals).to(utils.device), first_traces_lengths, second_traces_lengths

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
	def __init__(self, method: Type[ABC], env_name: str, problems: List[str],  train_configs: List, goal_to_task_str:MethodType, task_str_to_goal:MethodType, goals_adaptation_sequence_generation_method, is_fragmented=True, is_inference_same_length_sequences=False, is_learn_same_length_sequences=False, collect_statistics=True, specified_rl_algorithm = None, gc_sequence_generation=False, gc_goal_set=None):
		assert len(train_configs) == len(problems), "There should be exploration rate for every problem."
		self.train_configs = train_configs
		self.problems = problems
		self.env_name = env_name
		self.rl_agents_method = method
		self.agents: List[ABC] = []
		self.is_fragmented = is_fragmented
		self.is_inference_same_length_sequences = is_inference_same_length_sequences
		self.is_learn_same_length_sequences = is_learn_same_length_sequences
		# if is_continuous: self.train_func = train_metric_model_cont; self.collate_func = collate_fn_cont
		self.train_func = train_metric_model; self.collate_func = collate_fn
		self.collect_statistics = collect_statistics
		self.goal_to_task_str = goal_to_task_str
		self.task_str_to_goal = task_str_to_goal
		self.specified_rl_algorithm = specified_rl_algorithm
		self.goals_adaptation_sequence_generation_method = goals_adaptation_sequence_generation_method
		if gc_sequence_generation:
			assert gc_goal_set != None
		self.gc_goal_set = gc_goal_set

	def domain_learning_phase(self, problem_list_to_str_tuple:MethodType, input_size, hidden_size, batch_size):
		# start by training each rl agent on the base goal set
		for problem_name, (exploration_rate, num_timesteps) in zip(self.problems, self.train_configs):
			kwargs = {"env_name":self.env_name, "problem_name":problem_name}
			if self.specified_rl_algorithm: kwargs["algorithm"] = self.specified_rl_algorithm
			if exploration_rate: kwargs["exploration_rate"] = exploration_rate
			if num_timesteps: kwargs["num_timesteps"] = num_timesteps
			agent = self.rl_agents_method(**kwargs)
			agent.learn()
			self.agents.append(agent)
		self.obs_space, self.preprocess_obss = utils.get_obss_preprocessor(self.agents[0].env.observation_space)
		# train the network so it will find a metric for the observations of the base agents such that traces of agents to different goals are far from one another		
		self.model_directory = get_model_dir(env_name=self.env_name, model_name=problem_list_to_str_tuple(self.problems), class_name=self.__class__.__name__)
		last_path = r"lstm_model"
		# if self.is_continuous : last_path += r"_cont"
		if self.is_fragmented : last_path += r"_fragmented"
		last_path += r".pth"
		self.model_file_path = os.path.join(self.model_directory, last_path)
		self.model = LstmObservations(input_size=input_size, hidden_size=hidden_size)
		self.model.to(utils.device)

		if os.path.exists(self.model_file_path):
			print(f"Loading pre-existing lstm model in {self.model_file_path}")
			load_weights(loaded_model=self.model, path=self.model_file_path)
		else:
			train_samples, dev_samples = generate_datasets(10000, self.agents, metrics.stochastic_amplified_selection, problem_list_to_str_tuple(self.problems), self.env_name, self.preprocess_obss, self.is_fragmented, self.is_learn_same_length_sequences, self.gc_goal_set)
			train_dataset = GRDataset(len(train_samples), train_samples)
			dev_dataset = GRDataset(len(dev_samples), dev_samples)
			self.train_func(self.model,	train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_func),
							dev_loader=DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_func))
			save_weights(model=self.model, path=self.model_file_path)

	def goals_adaptation_phase(self, new_goals):
		self.current_goals = new_goals
		# start by training each rl agent on the base goal set
		problems_goals = [(self.goal_to_task_str(tuply),tuply) for tuply in new_goals]
		self.embeddings_dict = {} # relevant if the embedding of the plan occurs during the goals adaptation phase
		self.plans_dict = {} # relevant if the embedding of the plan occurs during the inference phase
		# will change, for now let an optimal agent give us the trace
		for problem_name, goal in problems_goals:
			if self.goals_adaptation_sequence_generation_method == MCTS_BASED:
				obs = mcts_model.plan(self.env_name, problem_name, self.task_str_to_goal(problem_name))
			elif self.goals_adaptation_sequence_generation_method == AGENT_BASED:
				kwargs = {"env_name":self.env_name, "problem_name":problem_name}
				if self.specified_rl_algorithm: kwargs["algorithm"] = self.specified_rl_algorithm
				if self.train_configs[0][0]: kwargs["exploration_rate"] = self.train_configs[0][0]
				if self.train_configs[0][1]: kwargs["num_timesteps"] = self.train_configs[0][1]
				agent = self.rl_agents_method(**kwargs)
				agent.learn()
				obs = agent.generate_observation(action_selection_method=metrics.greedy_selection, random_optimalism=False, save_fig=False)
			elif self.goals_adaptation_sequence_generation_method == GC_AGENT_BASED:
				kwargs = {"env_name":self.env_name, "problem_name":problem_name}
				if self.specified_rl_algorithm: kwargs["algorithm"] = self.specified_rl_algorithm
				if self.train_configs[0][0]: kwargs["exploration_rate"] = self.train_configs[0][0]
				if self.train_configs[0][1]: kwargs["num_timesteps"] = self.train_configs[0][1]
				agent = self.rl_agents_method(**kwargs)
				agent.learn()
				obs = agent.generate_observation_gc(action_selection_method=metrics.greedy_selection, random_optimalism=False, save_fig=False, goal_idx=int(goal))
			else:
				raise TypeError("goals_adaptation_sequence_generation_method must be either AGENT_BASED or MCTS_BASED")

			if self.is_inference_same_length_sequences:
				self.plans_dict[goal] = obs
				continue
			# if got here, the embedding occurs during goals adaptation phase - faster but doesn't allow adjusting plan length after receiving the sequence at inference time.
			partial_obs = random_subset_with_order(obs, (int)(random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) * len(obs)), self.is_fragmented)
			# if self.is_continuous: embedding = self.model.embed_sequence_cont(partial_obs, self.preprocess_obss)
			embedding = self.model.embed_sequence(partial_obs)
			self.embeddings_dict[goal] = embedding
   
	def get_goal_plan(self, goal):
		assert self.plans_dict, "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't return the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
		return self.plans_dict[goal]

	def inference_phase(self, inf_sequence, true_goal, percentage):
		# Arrange storage
		embeddings_path = get_embeddings_result_path(self.env_name)
		if not os.path.exists(embeddings_path):
			os.makedirs(embeddings_path)

		# In case of adjusting plans length before embedding them, we embed them only now in the inference phase.
		if self.is_inference_same_length_sequences:
			assert self.plans_dict, "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't embed the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
			for goal, seq in self.plans_dict.items():
				partial_obs = random_subset_with_order(seq, len(inf_sequence), self.is_fragmented)
				# if self.is_continuous: embedding = self.model.embed_sequence_cont(partial_obs, self.preprocess_obss)
				simplified_partial_obs = self.agents[0].simplify_observation(partial_obs)
				embedding = self.model.embed_sequence(simplified_partial_obs)
				self.embeddings_dict[goal] = embedding
		else:
			assert self.embeddings_dict, "embeddings_dict wasn't created during goals_adaptation_phase and now inference phase can't use the embeddings. when inference_diff_length, keep the embeddings and not their plans during goals_adaptation_phase."
		# if self.is_continuous: new_embedding = self.model.embed_sequence_cont(inf_sequence, self.preprocess_obss)
		simplified_inf_sequence = self.agents[0].simplify_observation(inf_sequence)
		new_embedding = self.model.embed_sequence(simplified_inf_sequence)
		closest_goal, greatest_similarity = None, 0
		for (goal, embedding) in self.embeddings_dict.items():
			curr_similarity = torch.exp(-torch.sum(torch.abs(embedding-new_embedding)))
			if curr_similarity > greatest_similarity:
				# print(f'new closest goal is: {goal}')
				closest_goal = goal
				greatest_similarity = curr_similarity

		self.embeddings_dict[f"{true_goal}_true"] = new_embedding
		if self.collect_statistics:
			with open(embeddings_path + f'/{true_goal}_{percentage}_embeddings_dict.pkl', 'wb') as embeddings_file:
				dill.dump(self.embeddings_dict, embeddings_file)
		self.embeddings_dict.pop(f"{true_goal}_true")

		return closest_goal