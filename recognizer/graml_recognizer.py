from abc import ABC
from dataset.gr_dataset import GRDataset, generate_datasets
from ml.base import RLAgent
from typing import List, Tuple, Type
from types import MethodType
import numpy as np
from torch.utils.data import DataLoader

from ml.sequential.lstm_model import LstmObservations, train_metric_model

### IMPLEMENT MORE SELECTION METHODS, MAKE SURE action_probs IS AS IT SEEMS: list of action-probability 'es ###
    

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
        train_samples, dev_samples = generate_datasets(10000, self.agents, agent.stochastic_softmax_selection)
        train_dataset = GRDataset(train_samples)
        dev_dataset = GRDataset(dev_samples)
        model = LstmObservations(10, max_episode_length=20)
        model = train_metric_model(model,
                                    train_loader=DataLoader(train_dataset, dev_dataset, batch_size=32, shuffle=False),
                                    dev_loader=DataLoader(dev_dataset, dev_dataset, batch_size=32, shuffle=False))

    def goals_adaptation_phase(self, new_goals):
        # start by training each rl agent on the base goal set
        for problem_name in self.problems:
            agent = self.rl_agents_method(env_name=self.env_name, problem_name=problem_name)
            agent.learn()
            self.agents.append(agent)
