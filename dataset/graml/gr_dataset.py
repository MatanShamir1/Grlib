import numpy as np
from torch.utils.data import Dataset
import random
from types import MethodType
from typing import List
from ml.utils import get_siamese_dataset_path
from ml.base import RLAgent
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

def generate_datasets(num_samples, agents: List[RLAgent], observation_creation_method : MethodType, problems: str, env_name, preprocess_obss, is_continuous=False, is_fragmented=True, is_learn_same_length_sequences=False):
    dataset_directory = get_siamese_dataset_path(env_name=env_name, problem_names=problems)
    if is_continuous: dataset_directory = os.path.join(dataset_directory, 'cont')
    if is_fragmented: addition = 'fragmented_'
    else: addition = ''
    dataset_train_path, dataset_dev_path = os.path.join(dataset_directory, addition + 'train.pkl'), os.path.join(dataset_directory, addition + 'dev.pkl')
    if os.path.exists(dataset_train_path) and os.path.exists(dataset_dev_path):
        print(f"Loading pre-existing datasets in {dataset_directory}")
        with open(dataset_train_path, 'rb') as train_file:
            train_samples = dill.load(train_file)
        with open(dataset_dev_path, 'rb') as dev_file:
            dev_samples = dill.load(dev_file)
    else:
        if not os.path.exists(dataset_directory):
            os.makedirs(dataset_directory)
        all_samples = []
        for i in range(num_samples):
            is_same_goal = (np.random.choice([1, 0], 1, p=[1/len(agents), 1 - 1/len(agents)]))[0]
            first_agent = np.random.choice(agents)
            first_trace_percentage = random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            first_observation = first_agent.generate_partial_observation(action_selection_method=observation_creation_method, percentage=first_trace_percentage, is_fragmented=is_fragmented)
            if is_continuous:
                first_observation = [preprocess_obss([obs])[0] for ((obs, (_, _)), _) in first_observation] # list of dicts, each dict a sample comprised of image and text
            else:
                first_observation = [(obs['direction'], agent_pos_x, agent_pos_y, action) for ((obs, (agent_pos_x, agent_pos_y)), action) in first_observation] # list of tuples, each tuple the sample
            second_agent = first_agent
            if not is_same_goal:
                second_agent = np.random.choice([agent for agent in agents if agent != first_agent])
            second_trace_percentage = first_trace_percentage if is_learn_same_length_sequences else random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            second_observation = second_agent.generate_partial_observation(action_selection_method=observation_creation_method, percentage=second_trace_percentage, is_fragmented=is_fragmented)
            if is_continuous:
                second_observation = [preprocess_obss([obs])[0] for ((obs, (_, _)), _) in second_observation]
            else:
                second_observation = [(obs['direction'], agent_pos_x, agent_pos_y, action) for ((obs, (agent_pos_x, agent_pos_y)), action) in second_observation]
            if not is_continuous: all_samples.append((
                [torch.tensor(observation, dtype=torch.float32) for observation in first_observation],
                [torch.tensor(observation, dtype=torch.float32) for observation in second_observation],
                torch.tensor(is_same_goal, dtype=torch.float32)))
            else: all_samples.append((first_observation, second_observation, torch.tensor(is_same_goal, dtype=torch.float32)))
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

    