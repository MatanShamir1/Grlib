from typing import Tuple
from env_wrappers import DiscreteActionFactor, ParkingDiscreteActionFactor, discretize_observation_space
from gymnasium.wrappers import TransformObservation
from minigrid.wrappers import FullyObsWrapper
import gymnasium
import panda_gym

EMPTY_STRING = ""
MINIGRID_ENV_PREFIX = "MiniGrid"
PANDA_DISCRETE_ENV_PREFIX = "DiscretePanda"
PARKING_DISCRETE_ENV_PREFIX = "DiscreteParking"
DISCRETE_ENV_NAME_SEPERATOR = "Y"
DISCRETE_FACTOR_SEPERATOR = "X"
DOT = "."
PARSING_ENV_NAME = (
    PANDA_DISCRETE_ENV_PREFIX,
    PARKING_DISCRETE_ENV_PREFIX
)


def panda_discrete_environment(
        env: gymnasium.Env,
        observation_discrete_factor: float,
        action_discrete_factor: float
) -> gymnasium.Env:
    env = TransformObservation(env, lambda obs: discretize_observation_space(obs, observation_discrete_factor))
    return DiscreteActionFactor(env, action_discrete_factor)


def parking_discrete_environment(
        env: gymnasium.Env,
        observation_discrete_factor: float,
        action_discrete_factor: float
) -> gymnasium.Env:
    env = TransformObservation(env, lambda obs: discretize_observation_space(obs, observation_discrete_factor))
    return ParkingDiscreteActionFactor(env, action_discrete_factor)


def parse_discrete_environment(
    env_name: str
) -> Tuple[float, str]:
    for env_prefix in PARSING_ENV_NAME:
        if env_name.startswith(env_prefix):
            env_name_str = env_name.replace(env_prefix, EMPTY_STRING)
            discrete_factor_str, env_name_str = env_name_str.split(DISCRETE_ENV_NAME_SEPERATOR)
            return (
                float(
                    discrete_factor_str.replace(DISCRETE_FACTOR_SEPERATOR, DOT)
                ),
                env_name_str
            )
    raise Exception(f"No matching parser for {env_name}, {PARSING_ENV_NAME}")


def make(
        env_name: str,
        observation_discrete_factor: float = 0.07,
        action_discrete_factor: float = 0.07
) -> gymnasium.Env:
    if env_name.startswith(MINIGRID_ENV_PREFIX):
        env = gymnasium.make(env_name)
        env = FullyObsWrapper(env)
    elif env_name.startswith(PANDA_DISCRETE_ENV_PREFIX):
        action_discrete_factor_1, env_name = parse_discrete_environment(env_name=env_name)
        env = gymnasium.make(env_name)
        env = panda_discrete_environment(
            env=env,
            observation_discrete_factor=observation_discrete_factor,
            action_discrete_factor=action_discrete_factor
        )
    elif env_name.startswith(PARKING_DISCRETE_ENV_PREFIX):
        action_discrete_factor_1, env_name = parse_discrete_environment(env_name=env_name)
        env = gymnasium.make(env_name)
        print(f"Creating an environment named:{env_name}!!!!!!!")
        env = parking_discrete_environment(
            env=env,
            observation_discrete_factor=observation_discrete_factor,
            action_discrete_factor=action_discrete_factor
        )
    else:
        env = gymnasium.make(env_name)
    return env


def get_actions_dimensions(env) -> int:
    action_space = env.action_space
    if isinstance(action_space, gymnasium.spaces.Box):
        return action_space.shape[0]
    elif isinstance(action_space, gymnasium.spaces.Discrete):
        print(f"action_space:{action_space} -> {action_space.n}")
        return action_space.n
    assert "Can't fetch the actions dimensions"
