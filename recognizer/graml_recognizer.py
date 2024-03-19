from abc import ABC
from dataset.gr_dataset import GRDataset, generate_datasets
from ml.base import RLAgent
from metrics import metrics
from typing import List, Tuple, Type
from types import MethodType
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

from ml.sequential.lstm_model import LstmObservations, train_metric_model

### IMPLEMENT MORE SELECTION METHODS, MAKE SURE action_probs IS AS IT SEEMS: list of action-probability 'es ###
    
def collate_fn(batch):
    first_traces, second_traces, is_same_goals = zip(*batch)
    first_traces_padded = pad_sequence(torch.tensor(first_traces), batch_first=True)
    second_traces_padded = pad_sequence(torch.tensor(second_traces), batch_first=True)
    first_traces_padded_lengths = [len(trace) for trace in first_traces]
    second_traces_padded_lengths = [len(trace) for trace in second_traces]
    return first_traces_padded, second_traces_padded, is_same_goals, first_traces_padded_lengths, second_traces_padded_lengths

class GramlRecognizer(ABC):
    def __init__(self, method: Type[RLAgent], env_name: str, problems: List[str], grid_size):
        self.problems = problems
        self.env_name = env_name
        self._observability_percentages = [0.1, 0.3, 0.5, 0.7, 1.0]
        self.rl_agents_method = method
        self.agents: List[RLAgent] = []
        
    def domain_learning_phase(self):
        # start by training each rl agent on the base goal set
        for problem_name in self.problems:
            agent = self.rl_agents_method(env_name=self.env_name, problem_name=problem_name)
            agent.learn()
            self.agents.append(agent)

        # train the network so it will find a metric for the observations of the base agents such that traces of agents to different goals are far from one another
        train_samples, dev_samples = generate_datasets(10000, self.agents, metrics.stochastic_amplified_selection, self.problems, self.env_name)
        train_dataset = GRDataset(len(train_samples), train_samples)
        dev_dataset = GRDataset(len(dev_samples), dev_samples)
        model = LstmObservations()
        model = train_metric_model(model,
                                    train_loader=DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn),
                                    dev_loader=DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn))

    def goals_adaptation_phase(self, new_goals):
        # start by training each rl agent on the base goal set
        for problem_name in self.problems:
            agent = self.rl_agents_method(env_name=self.env_name, problem_name=problem_name)
            agent.learn()
            self.agents.append(agent)
