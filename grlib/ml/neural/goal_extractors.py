from grlib.environment.environment import SupportedEnvs

from gr_libs.highway_env_scripts.envs.parking_env import ParkingEnv
from gr_libs.panda_gym_scripts.envs import PandaMyReachEnv
import gymnasium
import numpy as np
import pandas as pd


def panda_goal_extraction(env: PandaMyReachEnv):
    return env.task.goal


def parking_goal_extraction(env: ParkingEnv):
    return np.ravel(
        pd.DataFrame.from_records([env.goal.to_dict()])[env.observation_type.features]
    ) / env.observation_type.scales


GOALS_EXTRACTORS = {
    SupportedEnvs.PandaGym.value: panda_goal_extraction,
    SupportedEnvs.Parking.value: parking_goal_extraction,
}


def extract_goal(env_name: str, env: gymnasium.Env):
    for obs_name, goal_extractor in GOALS_EXTRACTORS.items():
        if obs_name in env_name.lower():
            return goal_extractor(env)

    raise Exception(f"Didn't find goal extractor for env named {env_name}, goal extractors:{GOALS_EXTRACTORS.items()}")
