import numpy as np
from torch.utils.data import Dataset
import random
from types import MethodType
from typing import List
from ml.utils import get_siamese_dataset_path
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

# this method fits both tabular and sequential according to generate_observation
def generate_datasets(num_samples, agents, observation_creation_method : MethodType, problems: List[str], env_name):
    dataset_directory = get_siamese_dataset_path(env_name=env_name, problem_names_list=problems)
    dataset_train_path, dataset_dev_path = os.path.join(dataset_directory, 'train.pkl'), os.path.join(dataset_directory, 'dev.pkl')
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
            is_same_goal = random.choice([1, 0]) # decide whether this sample constitutes 2 sequences to the same goal or to different goals
            first_agent = np.random.choice(agents) # pick a random agent to generate the sample
            first_observation = first_agent.generate_observation(observation_creation_method)
            first_observation = [(obs['direction'], agent_pos_x, agent_pos_y, action) for ((obs, (agent_pos_x, agent_pos_y)), action) in first_observation]
            second_agent = first_agent
            if not is_same_goal:
                second_agent = np.random.choice([agent for agent in agents if agent != first_agent]) # pick another random agent to generate the sample
            second_observation = second_agent.generate_observation(observation_creation_method)
            second_observation = [(obs['direction'], agent_pos_x, agent_pos_y, action) for ((obs, (agent_pos_x, agent_pos_y)), action) in second_observation]
            all_samples.append(([torch.tensor(observation, dtype=torch.float32) for observation in first_observation],
                                [torch.tensor(observation, dtype=torch.float32) for observation in second_observation],
                                torch.tensor(is_same_goal, dtype=torch.float32))) # observations to the same goal have label 1, and to different have label 0
            if i % 1000 == 0: print(f'generated {i} samples')
        # Split samples into train and dev sets
        total_samples = len(all_samples)
        train_size = int(0.8 * total_samples)
        train_samples = all_samples[:train_size]
        dev_samples = all_samples[train_size:]
        with open(dataset_train_path, 'wb') as train_file:
            dill.dump(train_samples, train_file)
        with open(dataset_dev_path, 'wb') as dev_file:
            dill.dump(dev_samples, dev_file)

    return train_samples, dev_samples

    