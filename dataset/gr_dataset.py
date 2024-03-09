import numpy as np
from torch.utils.data import Dataset
import random
from types import MethodType

class GRDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.samples = []

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx] # returns a tuple - as appended in 'generate_dataset' last line


def generate_datasets(self, num_samples, agents, observation_creation_method : MethodType):
    all_samples = []
    for _ in range(num_samples):
        is_same_goal = random.choice([True, False]) # decide whether this sample constitutes 2 sequences to the same goal or to different goals
        first_agent = np.random.choice(agents) # pick a random agent to generate the sample
        first_observation = first_agent.generate_observation(observation_creation_method)
        second_agent = first_agent
        if not is_same_goal:
            second_agent = np.random.choice([agent for agent in self.agents if agent != first_agent]) # pick another random agent to generate the sample
        second_observation = second_agent.generate_observation(observation_creation_method)
        all_samples.append((first_observation, second_observation, is_same_goal)) # observations to the same goal have label 1, and to different have label 0
    
    # Split samples into train and dev sets
    total_samples = len(all_samples)
    train_size = int(0.8 * total_samples)
    train_samples = all_samples[:train_size]
    dev_samples = all_samples[train_size:]

    return train_samples, dev_samples

    