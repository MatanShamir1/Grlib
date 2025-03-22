import numpy as np
from torch.utils.data import Dataset
import random
from types import MethodType
from typing import List
from gr_libs.environment.environment import EnvProperty
from gr_libs.metrics.metrics import measure_average_sequence_distance
from gr_libs.ml.base.rl_agent import ContextualAgent
from gr_libs.ml.utils import get_siamese_dataset_path
from gr_libs.ml.base import RLAgent
import os
import dill
import torch

class GRDataset(Dataset):
	def __init__(self, num_samples, samples):
		self.num_samples = num_samples
		self.samples = samples

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		return self.samples[idx] # returns a tuple - as appended in 'generate_dataset' last line

def check_diff_goals(first_agent_goal, second_agent_goal):
	try:
		assert first_agent_goal != second_agent_goal
	except Exception as e:
		try:
			assert any(first_agent_goal != second_agent_goal)
		except Exception as e:
			for arr1, arr2 in zip(first_agent_goal, second_agent_goal):
				assert any(elm1!=elm2 for elm1, elm2 in zip(arr1, arr2))

def generate_datasets(num_samples, agents: List[ContextualAgent], observation_creation_method : MethodType, problems: List[str], env_prop:EnvProperty, recognizer_name:str, gc_goal_set=None):
	if gc_goal_set: model_name = env_prop.name
	else: model_name = env_prop.problem_list_to_str_tuple(problems)
	dataset_directory = get_siamese_dataset_path(domain_name=env_prop.domain_name, env_name=env_prop.name, model_name=model_name, recognizer=recognizer_name)
	dataset_train_path, dataset_dev_path = os.path.join(dataset_directory, 'train.pkl'), os.path.join(dataset_directory, 'dev.pkl')
	if os.path.exists(dataset_train_path) and os.path.exists(dataset_dev_path):
		print(f"Loading pre-existing datasets in {dataset_directory}")
		with open(dataset_train_path, 'rb') as train_file:
			train_samples = dill.load(train_file)
		with open(dataset_dev_path, 'rb') as dev_file:
			dev_samples = dill.load(dev_file)
	else:
		print(f"{dataset_directory} doesn't exist, generating datasets")
		if not os.path.exists(dataset_directory):
			os.makedirs(dataset_directory)
		all_samples = []
		for i in range(num_samples):
			if gc_goal_set != None: # TODO change to having one flow for both cases and injecting according to gc_goal_set or not
				assert env_prop.gc_adaptable() == True, "shouldn't specify a goal directed representation if not generating datasets with a general agent."
				is_same_goal = (np.random.choice([1, 0], 1, p=[1/max(len(gc_goal_set), 6), 1 - 1/max(len(gc_goal_set), 6)]))[0]
				first_is_consecutive = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
				first_random_index = np.random.randint(0, len(gc_goal_set)) # works for lists of every object type, while np.choice only works for 1d arrays
				first_agent_goal = gc_goal_set[first_random_index] # could be either a real goal or a goal-directed problem name
				#first_agent_goal = np.random.choice(gc_goal_set)
				first_trace_percentage = random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
				first_observation = []
				first_agent_kwargs = {
					"action_selection_method": observation_creation_method,
					"percentage": first_trace_percentage,
					"is_consecutive": first_is_consecutive,
					"save_fig": False
				}
				while first_observation == []:
					# needs to be different than agents[0] problem_name, it should be from the gc_goal_set.
					# but the problem is with the panda because it
					if env_prop.use_goal_directed_problem(): first_agent_kwargs["goal_directed_problem"] = first_agent_goal
					else: first_agent_kwargs["goal_directed_goal"] = first_agent_goal
					first_observation = agents[0].agent.generate_partial_observation(**first_agent_kwargs)
				first_observation = agents[0].agent.simplify_observation(first_observation)

				second_is_consecutive = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
				second_agent_goal = first_agent_goal
				second_random_index = first_random_index
				if not is_same_goal:
					second_random_index = np.random.choice([i for i in range(len(gc_goal_set)) if i != first_random_index])
					assert first_random_index != second_random_index
				second_agent_goal = gc_goal_set[second_random_index]
				if not is_same_goal: check_diff_goals(first_agent_goal, second_agent_goal)
				second_trace_percentage = first_trace_percentage
				second_observation = []
				second_agent_kwargs = {
					"action_selection_method": observation_creation_method,
					"percentage": second_trace_percentage,
					"is_consecutive": second_is_consecutive,
					"save_fig": False
				}
				while second_observation == []:
					if env_prop.use_goal_directed_problem() == True: second_agent_kwargs["goal_directed_problem"] = second_agent_goal
					else: second_agent_kwargs["goal_directed_goal"] = second_agent_goal
					second_observation = agents[0].agent.generate_partial_observation(**second_agent_kwargs)
				second_observation = agents[0].agent.simplify_observation(second_observation)
			else:
				is_same_goal = (np.random.choice([1, 0], 1, p=[1/max(len(agents), 6), 1 - 1/max(len(agents), 6)]))[0]
				first_is_consecutive = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
				first_agent = np.random.choice(agents)
				first_trace_percentage = random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
				first_observation = first_agent.agent.generate_partial_observation(action_selection_method=observation_creation_method, percentage=first_trace_percentage, is_consecutive=first_is_consecutive, save_fig=False, random_optimalism=True)
				first_observation = first_agent.agent.simplify_observation(first_observation)
				
				second_agent = first_agent
				if not is_same_goal:
					second_agent = np.random.choice([agent for agent in agents if agent != first_agent])
					assert second_agent != first_agent
				second_is_consecutive = np.random.choice([True, False], 1, p=[0.5, 0.5])[0]
				second_trace_percentage = first_trace_percentage
				second_observation = second_agent.agent.generate_partial_observation(action_selection_method=observation_creation_method, percentage=second_trace_percentage, is_consecutive=second_is_consecutive, save_fig=False, random_optimalism=True)
				second_observation = second_agent.agent.simplify_observation(second_observation)
				if is_same_goal:
					observations_distance = measure_average_sequence_distance(first_observation, second_observation) # for debugging mate
			all_samples.append((
				[torch.tensor(observation, dtype=torch.float32) for observation in first_observation],
				[torch.tensor(observation, dtype=torch.float32) for observation in second_observation],
				torch.tensor(is_same_goal, dtype=torch.float32)))
			# all_samples.append((first_observation, second_observation, torch.tensor(is_same_goal, dtype=torch.float32)))
			if i % 1000 == 0:
				print(f'generated {i} samples')

		total_samples = len(all_samples)
		train_size = int(0.8 * total_samples)
		train_samples = all_samples[:train_size]
		dev_samples = all_samples[train_size:]
		with open(dataset_train_path, 'wb') as train_file:
			dill.dump(train_samples, train_file)
		with open(dataset_dev_path, 'wb') as dev_file:
			dill.dump(dev_samples, dev_file)

	return train_samples, dev_samples

	