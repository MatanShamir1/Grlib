from abc import ABC, abstractmethod
from typing import Any

State = Any


class ContextualAgent:
    """
    A class representing a contextual agent for reinforcement learning, including gym properties.

    Args:
        problem_name (str): The name of the problem the agent is designed to solve.
        problem_goal (str): The goal of the problem the agent is designed to achieve.
        agent: The underlying agent implementation.

    Attributes:
        problem_name (str): The name of the problem the agent is designed to solve.
        problem_goal (str): The goal of the problem the agent is designed to achieve.
        agent: The underlying agent implementation.
    """

    def __init__(self, problem_name, problem_goal, agent):
        """
        Initializes a reinforcement learning agent.

        Args:
            problem_name (str): The name of the problem.
            problem_goal (str): The goal of the problem.
            agent: The agent object.
        """
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
        domain_name: str,
    ):
        """
        Initializes a reinforcement learning agent.

        Args:
            episodes (int): The number of episodes to train the agent.
            decaying_eps (bool): Whether to use decaying epsilon-greedy exploration.
            epsilon (float): The exploration rate.
            learning_rate (float): The learning rate.
            gamma (float): The discount factor.
            problem_name (str): The name of the problem.
            domain_name (str): The name of the domain.
        """
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
        """
        Abstract method for the agent to learn from the environment.
        """

    def class_name(self):
        """
        Returns the name of the agent's class.

        Returns:
            str: The name of the agent's class.
        """
        return self.__class__.__name__

    def get_actions_probabilities(self, observation):
        """
        Get the probabilities of available actions given an observation.

        Args:
            observation: The observation from the environment.

        Raises:
            Exception: This function is unimplemented.

        Returns:
            Any: The probabilities of available actions.
        """
        raise Exception("function get_actions_probabilities is unimplemented")

    def get_number_of_unique_states(self):
        """
        Get the number of unique states encountered by the agent.

        Returns:
            int: The number of unique states encountered.
        """
        return len(self.states_counter)

    def update_states_counter(self, observation_str: str):
        """
        Update the counter for the number of times each observation state is encountered.

        Args:
            observation_str (str): The string representation of the observation state.
        """
        if observation_str in self.states_counter:
            self.states_counter[observation_str] = (
                self.states_counter[observation_str] + 1
            )
        else:
            self.states_counter[observation_str] = 1
        if len(self.states_counter) % 10000 == 0:
            print(f"probably error to many {len(self.states_counter)}")
