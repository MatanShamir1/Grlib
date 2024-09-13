import collections
from typing import Sequence, List
from gymnasium.spaces import Box
from gymnasium.wrappers import TransformObservation
import itertools
import gymnasium
import numpy as np


def count_after_dot(n: float) -> int:
    return str(n)[::-1].find('.')


def modify_minus_zero_obs(x):
    if type(x) == int or type(x) == float or type(x) == np.float64:
        return 0 if x == 0 else x
    elif type(x) is list or type(x) is np.ndarray:
        return np.array([0 if arg == 0 else arg for arg in x])
    raise Exception(f"Unsupported type of obs :{type(x)}")


def round_by_base(x, base):
    # in case x is -0.0
    x = modify_minus_zero_obs(x)
    prec = count_after_dot(base)
    return np.round(base * np.round(x/base), prec)


def discretize_observation_space(obs, discretization_factor: float):
    if type(obs) is dict or type(obs) is collections.OrderedDict:
        return {
            key: round_by_base(observation_array, discretization_factor)
            for key, observation_array in obs.items()
        }
    if type(obs) is np.ndarray:
        print("obs is nd array")
        return (obs / discretization_factor).astype(int)
    raise Exception(f"Invalid observation, type:{type(obs)}")


class DiscreteActionFactor(gymnasium.ActionWrapper, gymnasium.utils.RecordConstructorArgs):
    def __init__(
            self,
            env: gymnasium.Env,
            discretization_factor: float
    ):
        assert isinstance(
            env.action_space, Box
        ), f"expected Box action space, got {type(env.action_space)}"

        gymnasium.utils.RecordConstructorArgs.__init__(
            self,
            discretization_factor=discretization_factor
        )
        gymnasium.ActionWrapper.__init__(self, env)

        self._discretization_factor = discretization_factor
        self._actions = self._discrete_action_space(env, discretization_factor)
        self.action_space = gymnasium.spaces.Discrete(len(self._actions))

    @staticmethod
    def _discrete_action_space(
            env: gymnasium.Env,
            discretization_factor: float
    ) -> List[List[float]]:
        epsilon = 0.000001
        low = env.action_space.low
        high = env.action_space.high

        low_x, low_y, low_z = low
        high_x, high_y, high_z = high

        x = [round_by_base(elem, discretization_factor) for elem in
             np.arange(low_x, high_x + epsilon, discretization_factor)]
        y = [round_by_base(elem, discretization_factor) for elem in
             np.arange(low_y, high_y + epsilon, discretization_factor)]
        z = [round_by_base(elem, discretization_factor) for elem in
             np.arange(low_z, high_z + epsilon, discretization_factor)]

        return [list(prod) for prod in itertools.product(*[x, y, z])]

    def action(self, action: int):
        act = self._actions[action]
        # print(f"action:{act}")
        return act


class ParkingDiscreteActionFactor(gymnasium.ActionWrapper, gymnasium.utils.RecordConstructorArgs):
    def __init__(
            self,
            env: gymnasium.Env,
            discretization_factor: float
    ):
        assert isinstance(
            env.action_space, Box
        ), f"expected Box action space, got {type(env.action_space)}"

        gymnasium.utils.RecordConstructorArgs.__init__(
            self,
            discretization_factor=discretization_factor
        )
        gymnasium.ActionWrapper.__init__(self, env)

        self._discretization_factor = discretization_factor
        self._actions = self._discrete_action_space(env, discretization_factor)
        self.action_space = gymnasium.spaces.Discrete(len(self._actions))

    @staticmethod
    def _discrete_action_space(
            env: gymnasium.Env,
            discretization_factor: float
    ) -> List[List[float]]:
        epsilon = 0.000001
        low = env.action_space.low
        high = env.action_space.high

        low_x, low_y = low
        high_x, high_y = high

        x = [round_by_base(elem, discretization_factor) for elem in
             np.arange(low_x, high_x + epsilon, discretization_factor)]
        y = [round_by_base(elem, discretization_factor) for elem in
             np.arange(low_y, high_y + epsilon, discretization_factor)]

        return [list(prod) for prod in itertools.product(*[x, y])]

    def action(self, action: int):
        act = self._actions[action]
        # print(f"action:{act}")
        return act
