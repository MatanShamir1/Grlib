from abc import ABC, abstractmethod
from collections import namedtuple
import os
from gr_libs.environment.environment import EnvProperty, GCEnvProperty, LSTMProperties
from gr_libs.ml import utils
from gr_libs.ml.base import ContextualAgent
from typing import List, Tuple
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent, GCDeepRLAgent
from gr_libs.ml.planner.mcts import mcts_model
import dill
from gr_libs.ml.tabular.tabular_q_learner import TabularQLearner
from gr_libs.recognizer.graml.gr_dataset import GRDataset, generate_datasets
from gr_libs.ml.sequential.lstm_model import LstmObservations, train_metric_model
from gr_libs.ml.utils.format import random_subset_with_order
from gr_libs.ml.utils.storage import get_and_create, get_lstm_model_dir, get_embeddings_result_path, get_policy_sequences_result_path
from gr_libs.metrics import metrics
from gr_libs.recognizer.recognizer import GaAdaptingRecognizer, GaAgentTrainerRecognizer, LearningRecognizer, Recognizer # import first, very dependent

### TODO IMPLEMENT MORE SELECTION METHODS, MAKE SURE action_probs IS AS IT SEEMS: list of action-probability 'es ###

def collate_fn(batch):
	first_traces, second_traces, is_same_goals = zip(*batch)
	# torch.stack takes tensor tuples (fixed size) and stacks them up in a matrix
	first_traces_padded = pad_sequence([torch.stack(sequence) for sequence in first_traces], batch_first=True)
	second_traces_padded = pad_sequence([torch.stack(sequence) for sequence in second_traces], batch_first=True)
	first_traces_lengths = [len(trace) for trace in first_traces]
	second_traces_lengths = [len(trace) for trace in second_traces]
	return first_traces_padded.to(utils.device), second_traces_padded.to(utils.device), torch.stack(is_same_goals).to(utils.device), first_traces_lengths, second_traces_lengths

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

class Graml(LearningRecognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.agents: List[ContextualAgent] = []
		self.train_func = train_metric_model; self.collate_func = collate_fn

	@abstractmethod
	def train_agents_on_base_goals(self, base_goals: List[str], train_configs: List):
		pass

	def domain_learning_phase(self, base_goals: List[str], train_configs: List):
		super().domain_learning_phase(base_goals, train_configs)
		self.train_agents_on_base_goals(base_goals, train_configs)
		# train the network so it will find a metric for the observations of the base agents such that traces of agents to different goals are far from one another		
		self.model_directory = get_lstm_model_dir(domain_name=self.env_prop.domain_name, env_name=self.env_prop.name , model_name=self.env_prop.problem_list_to_str_tuple(self.original_problems), recognizer=self.__class__.__name__)
		last_path = r"lstm_model.pth"
		self.model_file_path = os.path.join(self.model_directory, last_path)
		self.model = LstmObservations(input_size=self.env_prop.get_lstm_props().input_size, hidden_size=self.env_prop.get_lstm_props().hidden_size)
		self.model.to(utils.device)

		if os.path.exists(self.model_file_path):
			print(f"Loading pre-existing lstm model in {self.model_file_path}")
			load_weights(loaded_model=self.model, path=self.model_file_path)
		else:
			print(f"{self.model_file_path} doesn't exist, training the model")
			train_samples, dev_samples = generate_datasets(num_samples=self.env_prop.get_lstm_props().num_samples,
												  		   agents=self.agents,
													   	   observation_creation_method=metrics.stochastic_amplified_selection,
														   problems=self.original_problems,
														   env_prop=self.env_prop,
														   gc_goal_set=self.gc_goal_set if hasattr(self, 'gc_goal_set') else None,
														   recognizer_name=self.__class__.__name__)

			train_dataset = GRDataset(len(train_samples), train_samples)
			dev_dataset = GRDataset(len(dev_samples), dev_samples)
			self.train_func(self.model,	train_loader=DataLoader(train_dataset, batch_size=self.env_prop.get_lstm_props().batch_size, shuffle=False, collate_fn=self.collate_func),
							dev_loader=DataLoader(dev_dataset, batch_size=self.env_prop.get_lstm_props().batch_size, shuffle=False, collate_fn=self.collate_func))
			save_weights(model=self.model, path=self.model_file_path)

	def goals_adaptation_phase(self, dynamic_goals: List[EnvProperty]):
		self.is_first_inf_since_new_goals = True
		self.current_goals = dynamic_goals
		# start by training each rl agent on the base goal set
		self.embeddings_dict = {} # relevant if the embedding of the plan occurs during the goals adaptation phase
		self.plans_dict = {} # relevant if the embedding of the plan occurs during the inference phase
		for goal in self.current_goals:
			obss = self.generate_sequences_library(goal)
			self.plans_dict[str(goal)] = obss

	def get_goal_plan(self, goal):
		assert self.plans_dict, "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't return the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
		return self.plans_dict[goal]

	def dump_plans(self, true_sequence, true_goal, percentage):
		assert self.plans_dict, "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't return the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
		# Arrange storage
		embeddings_path = get_and_create(get_embeddings_result_path(domain_name=self.env_prop.domain_name, env_name=self.env_prop.name, recognizer=self.__class__.__name__))
		self.plans_dict[f"{true_goal}_true"] = true_sequence

		with open(embeddings_path + f'/{true_goal}_{percentage}_plans_dict.pkl', 'wb') as plans_file:
			to_dump = {}
			for goal, obss in self.plans_dict.items():
				if goal == f"{true_goal}_true":
					to_dump[goal] = self.agents[0].agent.simplify_observation(obss)
				else:
					to_dump[goal] = []
					for obs in obss:
						addition = self.agents[0].agent.simplify_observation(obs) if self.is_first_inf_since_new_goals else obs
						to_dump[goal].append(addition)
			dill.dump(to_dump, plans_file)
		self.plans_dict.pop(f"{true_goal}_true")

	def create_embeddings_dict(self):
		for goal, obss in self.plans_dict.items():
			self.embeddings_dict[goal] = []
			for (cons_seq, non_cons_seq) in obss:
				self.embeddings_dict[goal].append((self.model.embed_sequence(cons_seq), self.model.embed_sequence(non_cons_seq)))

	def inference_phase(self, inf_sequence, true_goal, percentage) -> str:
		embeddings_path = get_and_create(get_embeddings_result_path(domain_name=self.env_prop.domain_name, env_name=self.env_prop.name, recognizer=self.__class__.__name__))
		simplified_inf_sequence = self.agents[0].agent.simplify_observation(inf_sequence)
		new_embedding = self.model.embed_sequence(simplified_inf_sequence)
		assert self.plans_dict, "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't embed the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
		if self.is_first_inf_since_new_goals:
			self.is_first_inf_since_new_goals = False
			self.update_sequences_library_inference_phase(inf_sequence)
			self.create_embeddings_dict()
		
		closest_goal, greatest_similarity = None, 0
		for (goal, embeddings) in self.embeddings_dict.items():
			sum_curr_similarities = 0
			for cons_embedding, non_cons_embedding in embeddings:
				sum_curr_similarities += max(torch.exp(-torch.sum(torch.abs(cons_embedding-new_embedding))), torch.exp(-torch.sum(torch.abs(non_cons_embedding-new_embedding))))
			mean_similarity = sum_curr_similarities/len(embeddings)
			if mean_similarity > greatest_similarity:
				closest_goal = goal
				greatest_similarity = mean_similarity

		self.embeddings_dict[f"{true_goal}_true"] = new_embedding
		if self.collect_statistics:
			with open(os.path.join(embeddings_path, f'{true_goal}_{percentage}_embeddings_dict.pkl'), 'wb') as embeddings_file:
				dill.dump(self.embeddings_dict, embeddings_file)
		self.embeddings_dict.pop(f"{true_goal}_true")

		return closest_goal
	
	@abstractmethod
	def generate_sequences_library(self, goal: str) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
		pass

	# this function duplicates every sequence and creates a consecutive and non-consecutive version of it
	def update_sequences_library_inference_phase(self, inf_sequence) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
		new_plans_dict = {}
		for goal, obss in self.plans_dict.items():
			new_obss = []
			for obs in obss:
				consecutive_partial_obs = random_subset_with_order(obs, len(inf_sequence), is_consecutive=True)
				non_consecutive_partial_obs = random_subset_with_order(obs, len(inf_sequence), is_consecutive=False)
				simplified_consecutive_partial_obs = self.agents[0].agent.simplify_observation(consecutive_partial_obs)
				simplified_non_consecutive_partial_obs = self.agents[0].agent.simplify_observation(non_consecutive_partial_obs)
				new_obss.append((simplified_consecutive_partial_obs, simplified_non_consecutive_partial_obs))
			new_plans_dict[goal] = new_obss # override old full observations with new partial observations with consecutive and non-consecutive versions.
		self.plans_dict = new_plans_dict

class BGGraml(Graml):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def domain_learning_phase(self, base_goals: List[str], train_configs: List):
		assert len(train_configs) == len(base_goals), "There should be train configs for every goal in BGGraml."
		return super().domain_learning_phase(base_goals, train_configs)
	
	# In case we need goal-directed agent for every goal
	def train_agents_on_base_goals(self, base_goals: List[str], train_configs: List):
		self.original_problems = [self.env_prop.goal_to_problem_str(g) for g in base_goals]
		# start by training each rl agent on the base goal set
		for (problem, goal), (algorithm, num_timesteps) in zip(zip(self.original_problems, base_goals), train_configs):
			kwargs = {"domain_name":self.domain_name, "problem_name":problem}
			if algorithm != None: kwargs["algorithm"] = algorithm
			if num_timesteps != None: kwargs["num_timesteps"] = num_timesteps
			agent = self.rl_agent_type(**kwargs)
			agent.learn()
			self.agents.append(ContextualAgent(problem_name=problem, problem_goal=goal, agent=agent))

class MCTSBasedGraml(BGGraml, GaAdaptingRecognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.rl_agent_type==None: self.rl_agent_type = TabularQLearner

	def generate_sequences_library(self, goal: str) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
		problem_name = self.env_prop.goal_to_problem_str(goal)
		img_path = os.path.join(get_policy_sequences_result_path(self.env_prop.domain_name, recognizer=self.__class__.__name__), problem_name + "_MCTS")
		return mcts_model.plan(self.env_prop.name, problem_name, goal, save_fig=True, img_path=img_path, env_prop=self.env_prop)

class ExpertBasedGraml(BGGraml, GaAgentTrainerRecognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.rl_agent_type==None:
			if self.env_prop.is_state_discrete() and self.env_prop.is_action_discrete():
				self.rl_agent_type = TabularQLearner
			else:
				self.rl_agent_type = DeepRLAgent

	def generate_sequences_library(self, goal: str) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
		problem_name = self.env_prop.goal_to_problem_str(goal)
		kwargs = {"domain_name":self.domain_name, "problem_name":problem_name}
		if self.dynamic_train_configs_dict[problem_name][0] != None: kwargs["algorithm"] = self.dynamic_train_configs_dict[problem_name][0]
		if self.dynamic_train_configs_dict[problem_name][1] != None: kwargs["num_timesteps"] = self.dynamic_train_configs_dict[problem_name][1]
		agent = self.rl_agent_type(**kwargs)
		agent.learn()
		fig_path = get_and_create(f"{os.path.abspath(os.path.join(get_policy_sequences_result_path(domain_name=self.env_prop.domain_name, env_name=self.env_prop.name, recognizer=self.__class__.__name__), problem_name))}_bg_sequence")
		return [agent.generate_observation(action_selection_method=metrics.greedy_selection, random_optimalism=False, save_fig=True, fig_path=fig_path, env_prop=self.env_prop)]

	def goals_adaptation_phase(self, dynamic_goals: List[str], dynamic_train_configs):
		self.dynamic_goals_problems = [self.env_prop.goal_to_problem_str(g) for g in dynamic_goals]
		self.dynamic_train_configs_dict = {problem:config for problem, config in zip(self.dynamic_goals_problems,dynamic_train_configs)}
		return super().goals_adaptation_phase(dynamic_goals)

class GCGraml(Graml, GaAdaptingRecognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if self.rl_agent_type==None: self.rl_agent_type = GCDeepRLAgent
		assert self.env_prop.gc_adaptable() and not self.env_prop.is_state_discrete() and not self.env_prop.is_action_discrete()

	def domain_learning_phase(self, base_goals: List[str], train_configs: List):
		assert len(train_configs) == 1, "There should be one train config for the sole gc agent in GCGraml."
		return super().domain_learning_phase(base_goals, train_configs)

	# In case we need goal-directed agent for every goal
	def train_agents_on_base_goals(self, base_goals: List[str], train_configs: List):
		self.gc_goal_set = base_goals
		self.original_problems = self.env_prop.name # needed for gr_dataset
		# start by training each rl agent on the base goal set
		kwargs = {"domain_name":self.domain_name, "problem_name":self.env_prop.name}
		algorithm, num_timesteps = train_configs[0] # should only be one, was asserted
		if algorithm != None: kwargs["algorithm"] = algorithm
		if num_timesteps != None: kwargs["num_timesteps"] = num_timesteps
		gc_agent = self.rl_agent_type(**kwargs)
		gc_agent.learn()
		self.agents.append(ContextualAgent(problem_name=self.env_prop.name, problem_goal="general", agent=gc_agent))

	def generate_sequences_library(self, goal: str) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
		problem_name = self.env_prop.goal_to_problem_str(goal)
		kwargs = {"domain_name":self.domain_name, "problem_name":self.env_prop.name} # problem name is env name in gc case
		if self.original_train_configs[0][0] != None: kwargs["algorithm"] = self.original_train_configs[0][0]
		if self.original_train_configs[0][1] != None: kwargs["num_timesteps"] = self.original_train_configs[0][1]
		agent = self.rl_agent_type(**kwargs)
		agent.learn()
		fig_path = get_and_create(f"{os.path.abspath(os.path.join(get_policy_sequences_result_path(domain_name=self.env_prop.domain_name, env_name=self.env_prop.name, recognizer=self.__class__.__name__), problem_name))}_gc_sequence")
		agent_kwargs = {
			"action_selection_method": metrics.stochastic_amplified_selection,
			"random_optimalism": True,
			"save_fig": True,
			"fig_path": fig_path
		}
		if self.env_prop.use_goal_directed_problem(): agent_kwargs["goal_directed_problem"] = problem_name
		else: agent_kwargs["goal_directed_goal"] = goal
		obss = []
		for _ in range(5): obss.append(agent.generate_observation(**agent_kwargs))
		return obss
