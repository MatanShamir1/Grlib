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

def generate_datasets(num_samples, agents: List, observation_creation_method : MethodType, problems: str, env_name, preprocess_obss, is_fragmented=True, is_learn_same_length_sequences=False, gc_goal_set=None):
    dataset_directory = get_siamese_dataset_path(env_name=env_name, problem_names=problems)
    # if is_continuous: dataset_directory = os.path.join(dataset_directory, 'cont')
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
            if gc_goal_set != None:
                is_same_goal = (np.random.choice([1, 0], 1, p=[1/len(gc_goal_set), 1 - 1/len(gc_goal_set)]))[0]
                first_agent_goal = np.random.choice(gc_goal_set)
                first_trace_percentage = random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                first_observation = agents[0].generate_partial_observation_gc(action_selection_method=observation_creation_method, percentage=first_trace_percentage, is_fragmented=is_fragmented, save_fig=False, goal_idx=first_agent_goal)
                first_observation = agents[0].simplify_observation(first_observation)
                second_agent_goal = first_agent_goal
                if not is_same_goal:
                    second_agent_goal = np.random.choice([goal for goal in gc_goal_set if goal != first_agent_goal])
                second_trace_percentage = second_trace_percentage if is_learn_same_length_sequences else random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                second_observation = agents[0].generate_partial_observation_gc(action_selection_method=observation_creation_method, percentage=second_trace_percentage, is_fragmented=is_fragmented, save_fig=False, goal_idx=second_agent_goal)
                second_observation = agents[0].simplify_observation(second_observation)
            else:
                is_same_goal = (np.random.choice([1, 0], 1, p=[1/len(agents), 1 - 1/len(agents)]))[0]
                first_agent = np.random.choice(agents)
                first_trace_percentage = random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                #first_agent.record_video("maze_video_live.mp4")
                first_observation = first_agent.generate_partial_observation(action_selection_method=observation_creation_method, percentage=first_trace_percentage, is_fragmented=is_fragmented, save_fig=False)
                    # first_observation = [preprocess_obss([obs])[0] for ((obs, (_, _)), _) in first_observation] # list of dicts, each dict a sample comprised of image and text
                first_observation = first_agent.simplify_observation(first_observation)
                second_agent = first_agent
                if not is_same_goal:
                    second_agent = np.random.choice([agent for agent in agents if agent != first_agent])
                second_trace_percentage = first_trace_percentage if is_learn_same_length_sequences else random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                second_observation = second_agent.generate_partial_observation(action_selection_method=observation_creation_method, percentage=second_trace_percentage, is_fragmented=is_fragmented, save_fig=False)
                    # second_observation = [preprocess_obss([obs])[0] for ((obs, (_, _)), _) in second_observation]
                second_observation = second_agent.simplify_observation(second_observation)
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

    