import gymnasium as gym
from abc import abstractmethod
from typing import Collection, Literal, Any
from random import Random
import numpy as np

from gr_libs.ml.base import RLAgent
from gr_libs.ml.base import State


class TabularRLAgent(RLAgent):
    """
    This is a base class used as parent class for any
    RL agent. This is currently not much in use, but is
    recommended as development goes on.
    """

    def __init__(self,
                 domain_name: str,
                 problem_name: str,
                 episodes: int,
                 decaying_eps: bool,
                 eps: float,
                 alpha: float,
                 decay: float,
                 gamma: float,
                 rand: Random,
                 learning_rate
                 ):
        super().__init__(
            episodes=episodes,
            decaying_eps=decaying_eps,
            epsilon=eps,
            learning_rate=learning_rate,
            gamma=gamma,
            domain_name=domain_name,
            problem_name=problem_name
        )
        self.env = gym.make(id=problem_name)
        self.actions = self.env.unwrapped.actions
        self.number_of_actions = len(self.actions)
        self._actions_space = self.env.action_space
        self._random = rand
        self._alpha = alpha
        self._decay = decay
        self._c_eps = eps
        self.q_table = {}

        # TODO:: maybe need to save env.reset output
        self.env.reset()

    @abstractmethod
    def agent_start(self, state) -> Any:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        pass

    @abstractmethod
    def agent_step(self, reward: float, state: State) -> Any:
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        pass

    @abstractmethod
    def agent_end(self, reward: float) -> Any:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass

    @abstractmethod
    def policy(self, state: State) -> Any:
        """The action for the specified state under the currently learned policy
           (unlike agent_step, this does not update the policy using state as a sample.
           Args:
                state (Any): the state observation from the environment
           Returns:
                The action prescribed for that state
        """
        pass

    @abstractmethod
    def softmax_policy(self, state: State) -> np.array:
        """Returns a softmax policy over the q-value returns stored in the q-table

        Args:
            state (State): the state for which we want a softmax policy

        Returns:
            np.array: probability of taking each action in self.actions given a state
        """
        pass

    @abstractmethod
    def learn(self, init_threshold: int = 20):
        pass

    def __getitem__(self, state: State) -> Any:
        """[summary]

        Args:
            state (Any): The state for which we want to get the policy

        Raises:
            InvalidAction: [description]

        Returns:
            Any: [description]
        """""
        return self.softmax_policy(state)
