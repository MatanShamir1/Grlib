from typing import Any
from abc import ABC, abstractmethod
import numpy as np

State = Any

class ContextualAgent:
    def __init__(self, problem_name, problem_goal, agent):
        self.problem_name = problem_name
        self.problem_goal = problem_goal
        self.agent = agent

class RLAgent(ABC):
    def __init__(
            self,
            episodes: int,
            decaying_eps: bool,
            epsilon: float,
            learning_rate: float,
            gamma: float,
            problem_name: str,
            domain_name: str
    ):
        self.episodes = episodes
        self.decaying_eps = decaying_eps
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.problem_name = problem_name
        self.domain_name = domain_name
        self.env = None
        self.states_counter = {}

    @abstractmethod
    def learn(self):
        pass

    def class_name(self):
        return self.__class__.__name__

    def get_actions_probabilities(self, observation):
        raise Exception("function get_actions_probabilities is unimplemented")

    def get_number_of_unique_states(self):
        return len(self.states_counter)

    def update_states_counter(self, observation_str: str):
        if observation_str in self.states_counter:
            self.states_counter[observation_str] = self.states_counter[observation_str] + 1
        else:
            self.states_counter[observation_str] = 1
        if len(self.states_counter) % 10000 == 0:
            print(f"probably error to many {len(self.states_counter)}")

