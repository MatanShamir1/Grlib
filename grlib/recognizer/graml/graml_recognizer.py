from abc import ABC
import math
import os
import random
from grlib import ml
from stable_baselines3 import SAC, TD3
from grlib.ml import utils
from grlib.ml.base import RLAgent, ContextualAgent
from typing import List, Tuple, Type
from types import MethodType
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from grlib.ml.planner.mcts import mcts_model
import dill
from grlib.recognizer.graml.gr_dataset import GRDataset, generate_datasets
from grlib.ml.sequential.lstm_model import LstmObservations, train_metric_model, train_metric_model_cont
from grlib.ml.utils.format import random_subset_with_order
from grlib.ml.utils.storage import get_lstm_model_dir, get_embeddings_result_path
from grlib.metrics import metrics # import first, very dependent

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
	def __init__(self, method: Type[ABC], env_name: str, problems: List[str],  train_configs: List,
              task_str_to_goal:MethodType, problem_list_to_str_tuple:MethodType, num_samples: int,
              goals_adaptation_sequence_generation_method, input_size: int, hidden_size: int, batch_size: int,
              partial_obs_type: str=True, specified_rl_algorithm_learning=None, specified_rl_algorithm_adaptation=None, 
              is_inference_same_length_sequences=False, is_learn_same_length_sequences=False,
              collect_statistics=True, gc_sequence_generation=False,
              gc_goal_set=None, tasks_to_complete: bool=False, use_goal_directed_problem=None):
		assert len(train_configs) == len(problems), "There should be train configs for every problem."
		self.train_configs = train_configs
		self.problems = problems
		self.env_name = env_name
		self.rl_agents_method = method
		self.agents: List[ContextualAgent] = []
		self.partial_obs_type = partial_obs_type
		self.is_inference_same_length_sequences = is_inference_same_length_sequences
		self.is_learn_same_length_sequences = is_learn_same_length_sequences
		# if is_continuous: self.train_func = train_metric_model_cont; self.collate_func = collate_fn_cont
		self.train_func = train_metric_model; self.collate_func = collate_fn
		self.collect_statistics = collect_statistics
		self.task_str_to_goal = task_str_to_goal
		self.specified_rl_algorithm_learning = specified_rl_algorithm_learning
		self.specified_rl_algorithm_adaptation = specified_rl_algorithm_adaptation
		self.goals_adaptation_sequence_generation_method = goals_adaptation_sequence_generation_method
		self.gc_sequence_generation = gc_sequence_generation
		if gc_sequence_generation:
			assert gc_goal_set != None
			assert use_goal_directed_problem != None # need to decide whether to use goal directed problem or specific goal to generate goal-directed traces using the general agent.
		self.gc_goal_set = gc_goal_set
		self.use_goal_directed_problem = use_goal_directed_problem
		self.tasks_to_complete = tasks_to_complete
		self.problem_list_to_str_tuple = problem_list_to_str_tuple
  
		# metric-model related fields
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.num_samples = num_samples
		

	def domain_learning_phase(self):
		# start by training each rl agent on the base goal set
		for problem_name, (exploration_rate, num_timesteps) in zip(self.problems, self.train_configs):
			kwargs = {"env_name":self.env_name, "problem_name":problem_name}
			if self.specified_rl_algorithm_learning: kwargs["algorithm"] = self.specified_rl_algorithm_learning
			if exploration_rate: kwargs["exploration_rate"] = exploration_rate
			if num_timesteps: kwargs["num_timesteps"] = num_timesteps
			if self.tasks_to_complete: kwargs["tasks_to_complete"] = [problem_name]; problem_name = self.env_name; kwargs["env_name"] = kwargs["env_name"][:-3] + "Env"; kwargs["problem_name"] = self.env_name; kwargs["complex_obs_space"] = True
			if self.gc_sequence_generation: kwargs["is_gc"] = True
			# edge case: not working for GI-14, GI-18, GI-15. TODO remove this by training them properly and changing configuration to be agent specific.
			# if problem_name in ["Parking-S-14-PC--GI-14-v0", "Parking-S-14-PC--GI-18-v0"]:
			# 	kwargs["algorithm"] = TD3
			if problem_name in ["Parking-S-14-PC--GI-15-v0", "Parking-S-14-PC--GI-13-v0", "Parking-S-14-PC--GI-20-v0"]:
				kwargs["algorithm"] = SAC
			agent = self.rl_agents_method(**kwargs)
			agent.learn()
			self.agents.append(ContextualAgent(problem_name=problem_name, problem_goal=self.task_str_to_goal(problem_name), agent=agent))
		self.obs_space, self.preprocess_obss = utils.get_obss_preprocessor(self.agents[0].agent.env.observation_space)
		# train the network so it will find a metric for the observations of the base agents such that traces of agents to different goals are far from one another		
		self.model_directory = get_lstm_model_dir(env_name=self.env_name, model_name=self.problem_list_to_str_tuple(self.problems))
		last_path = r"lstm_model_"
		# if self.is_continuous : last_path += r"_cont"
		last_path += self.partial_obs_type
		last_path += r".pth"
		self.model_file_path = os.path.join(self.model_directory, last_path)
		self.model = LstmObservations(input_size=self.input_size, hidden_size=self.hidden_size)
		self.model.to(utils.device)

		if os.path.exists(self.model_file_path):
			print(f"Loading pre-existing lstm model in {self.model_file_path}")
			load_weights(loaded_model=self.model, path=self.model_file_path)
		else:
			train_samples, dev_samples = generate_datasets(num_samples=self.num_samples,
                                                  		   agents=self.agents,
                                                       	   observation_creation_method=metrics.stochastic_amplified_selection,
                                                           problems=self.problem_list_to_str_tuple(self.problems),
                                                           env_name=self.env_name,
                                                           preprocess_obss=self.preprocess_obss,
                                                           is_fragmented=self.partial_obs_type=="fragmented",
                                                           is_learn_same_length_sequences=self.is_learn_same_length_sequences,
                                                           gc_goal_set=self.gc_goal_set,
                                                           use_goal_directed_problem=self.use_goal_directed_problem)

			train_dataset = GRDataset(len(train_samples), train_samples)
			dev_dataset = GRDataset(len(dev_samples), dev_samples)
			self.train_func(self.model,	train_loader=DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_func),
							dev_loader=DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_func))
			save_weights(model=self.model, path=self.model_file_path)

	def goals_adaptation_phase(self, dynamic_goals_problems, dynamic_train_configs):
		self.current_goals = [self.task_str_to_goal(problem) for problem in dynamic_goals_problems]
		# start by training each rl agent on the base goal set
		self.embeddings_dict = {} # relevant if the embedding of the plan occurs during the goals adaptation phase
		self.plans_dict = {} # relevant if the embedding of the plan occurs during the inference phase
		# will change, for now let an optimal agent give us the trace
		for i, (problem_name, goal) in enumerate(zip(dynamic_goals_problems, self.current_goals)):
			if self.goals_adaptation_sequence_generation_method == MCTS_BASED:
				obs = mcts_model.plan(self.env_name, problem_name, self.task_str_to_goal(problem_name), save_fig=True)
			elif self.goals_adaptation_sequence_generation_method == AGENT_BASED:
				kwargs = {"env_name":self.env_name, "problem_name":problem_name}
				if self.specified_rl_algorithm_adaptation: kwargs["algorithm"] = self.specified_rl_algorithm_adaptation
				# edge case: not working for GI-14, GI-18, GI-15. TODO remove this by training them properly and changing configuration to be agent specific.
				# if problem_name in ["Parking-S-14-PC--GI-14-v0", "Parking-S-14-PC--GI-18-v0"]:
				# 	kwargs["algorithm"] = TD3
				if problem_name in ["Parking-S-14-PC--GI-15-v0"]:
					kwargs["algorithm"] = SAC
				if dynamic_train_configs[i][0]!=None: kwargs["exploration_rate"] = dynamic_train_configs[i][0]
				if dynamic_train_configs[i][1]!=None: kwargs["num_timesteps"] = dynamic_train_configs[i][1]
				agent = self.rl_agents_method(**kwargs)
				agent.learn()
				obs = agent.generate_observation(action_selection_method=metrics.greedy_selection, random_optimalism=False, save_fig=True)
			elif self.goals_adaptation_sequence_generation_method == GC_AGENT_BASED:
				kwargs = {"env_name":self.env_name, "problem_name":self.problems[0]}
				if self.specified_rl_algorithm_adaptation: kwargs["algorithm"] = self.specified_rl_algorithm_adaptation
				if self.train_configs[0][0]: kwargs["exploration_rate"] = self.train_configs[0][0]
				if self.train_configs[0][1]: kwargs["num_timesteps"] = self.train_configs[0][1]
				kwargs["is_gc"] = True
				agent = self.rl_agents_method(**kwargs)
				agent.learn()
				agent_kwargs = {
					"action_selection_method": metrics.stochastic_amplified_selection,
					"random_optimalism": True,
					"save_fig": True
				}
				if self.use_goal_directed_problem == True: agent_kwargs["goal_directed_problem"] = problem_name
				elif self.use_goal_directed_problem == False: agent_kwargs["goal_directed_goal"] = goal # why not 'goal'? are they the same? check if that's the case for panda.
				obs = []
				for i in range(5):
					obs.append(agent.generate_observation_gc(**agent_kwargs))
			else:
				raise TypeError("goals_adaptation_sequence_generation_method must be either AGENT_BASED or MCTS_BASED")

			if self.is_inference_same_length_sequences: # ELSE IS PROBABLY BROKEN
				self.plans_dict[str(goal)] = obs
				continue
			# if got here, the embedding occurs during goals adaptation phase - faster but doesn't allow adjusting plan length after receiving the sequence at inference time.
			partial_obs = random_subset_with_order(obs, (int)(random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) * len(obs)), is_fragmented=self.partial_obs_type=="fragmented")
			simplified_partial_obs = self.agents[0].agent.simplify_observation(partial_obs)
			embedding = self.model.embed_sequence(simplified_partial_obs)
			self.embeddings_dict[str(goal)] = embedding
   
	def get_goal_plan(self, goal):
		assert self.plans_dict, "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't return the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
		return self.plans_dict[goal]

	def dump_plans(self, true_sequence, true_goal, percentage):
		assert self.plans_dict, "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't return the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
		# Arrange storage
		embeddings_path = get_embeddings_result_path(self.env_name)
		if not os.path.exists(embeddings_path):
			os.makedirs(embeddings_path)
		self.plans_dict[f"{true_goal}_true"] = true_sequence

		with open(embeddings_path + f'/{true_goal}_{percentage}_plans_dict.pkl', 'wb') as plans_file:
			if self.goals_adaptation_sequence_generation_method == GC_AGENT_BASED:
				to_dump = {}
				for goal, obss in self.plans_dict.items():
					if goal == f"{true_goal}_true":
						to_dump[goal] = self.agents[0].agent.simplify_observation(obss)
					else:
						to_dump[goal] = []
						for obs in obss:
							to_dump[goal].append(self.agents[0].agent.simplify_observation(obs))
			else:
				to_dump = {goal:self.agents[0].agent.simplify_observation(obs) for goal, obs in self.plans_dict.items()}
			dill.dump(to_dump, plans_file)
		self.plans_dict.pop(f"{true_goal}_true")
# python experiments.py --recognizer graml --domain parking --task L5 --partial_obs_type fragmented --parking_env gc_agent --collect_stats --inference_same_seq_len
	def inference_phase(self, inf_sequence, true_goal, percentage) -> str:
		# Arrange storage
		embeddings_path = get_embeddings_result_path(self.env_name)
		if not os.path.exists(embeddings_path):
			os.makedirs(embeddings_path)
   
		simplified_inf_sequence = self.agents[0].agent.simplify_observation(inf_sequence)
		new_embedding = self.model.embed_sequence(simplified_inf_sequence)

		# In case of adjusting plans length before embedding them, we embed them only now in the inference phase.
		if self.is_inference_same_length_sequences:
			assert self.plans_dict, "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't embed the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
			if self.goals_adaptation_sequence_generation_method in [AGENT_BASED, MCTS_BASED]:
				for goal, seq in self.plans_dict.items():
					partial_obs = random_subset_with_order(seq, len(inf_sequence), is_fragmented=self.partial_obs_type=="fragmented")
					# if self.is_continuous: embedding = self.model.embed_sequence_cont(partial_obs, self.preprocess_obss)
					simplified_partial_obs = self.agents[0].agent.simplify_observation(partial_obs)
					embedding = self.model.embed_sequence(simplified_partial_obs)
					self.embeddings_dict[goal] = embedding
					# if self.is_continuous: new_embedding = self.model.embed_sequence_cont(inf_sequence, self.preprocess_obss)
					closest_goal, greatest_similarity = None, 0
					for (goal, embedding) in self.embeddings_dict.items():
						curr_similarity = torch.exp(-torch.sum(torch.abs(embedding-new_embedding)))
						if curr_similarity > greatest_similarity:
							# print(f'new closest goal is: {goal}')
							closest_goal = goal
							greatest_similarity = curr_similarity
			else: # GC_AGENT_BASED
				for goal, obss in self.plans_dict.items():
					self.embeddings_dict[goal] = []
					for seq in obss:
						partial_obs = random_subset_with_order(seq, len(inf_sequence), is_fragmented=self.partial_obs_type=="fragmented")
						# if self.is_continuous: embedding = self.model.embed_sequence_cont(partial_obs, self.preprocess_obss)
						simplified_partial_obs = self.agents[0].agent.simplify_observation(partial_obs)
						embedding = self.model.embed_sequence(simplified_partial_obs)
						self.embeddings_dict[goal].append(embedding)
					closest_goal, greatest_similarity = None, 0
					for (goal, embeddings) in self.embeddings_dict.items():
						sum_curr_similarities = 0
						for embedding in embeddings:
							sum_curr_similarities += torch.exp(-torch.sum(torch.abs(embedding-new_embedding)))
						mean_similarity = sum_curr_similarities/len(embeddings)
						if mean_similarity > greatest_similarity:
							# print(f'new closest goal is: {goal}')
							closest_goal = goal
							greatest_similarity = mean_similarity
		else:
			assert self.embeddings_dict, "embeddings_dict wasn't created during goals_adaptation_phase and now inference phase can't use the embeddings. when inference_diff_length, keep the embeddings and not their plans during goals_adaptation_phase."
		

		self.embeddings_dict[f"{true_goal}_true"] = new_embedding
		if self.collect_statistics:
			with open(embeddings_path + f'/{true_goal}_{percentage}_embeddings_dict.pkl', 'wb') as embeddings_file:
				dill.dump(self.embeddings_dict, embeddings_file)
		self.embeddings_dict.pop(f"{true_goal}_true")

		return closest_goal