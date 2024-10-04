import os
import torch
import numpy as np
import gymnasium
from ml.neural.goal_extractors import extract_goal
from typing import Optional

from ml.base import RLAgent
from stable_baselines3 import SAC, PPO
from rl_zoo3.utils import get_model_path, get_latest_run_id
from collections import OrderedDict
from rl_zoo3 import create_test_env
from ml.utils import storage

NETWORK_SETUP = {
    "sac": OrderedDict([('batch_size', 512), ('buffer_size', 100000), ('ent_coef', 'auto'), ('gamma', 0.95), ('learning_rate', 0.001), ('learning_starts', 5000), ('n_timesteps', 50000.0), ('normalize', "{'norm_obs': False, 'norm_reward': False}"), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(net_arch=[64, 64])'), ('replay_buffer_class', 'HerReplayBuffer'), ('replay_buffer_kwargs', "dict( goal_selection_strategy='future', n_sampled_goal=4 )"), ('normalize_kwargs', {'norm_obs': False, 'norm_reward': False})]),
    "tqc": OrderedDict([('batch_size', 256), ('buffer_size', 1000000), ('ent_coef', 'auto'), ('env_wrapper', ['sb3_contrib.common.wrappers.TimeFeatureWrapper']), ('gamma', 0.95), ('learning_rate', 0.001), ('learning_starts', 1000), ('n_timesteps', 25000.0), ('normalize', False), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(net_arch=[64, 64])'), ('replay_buffer_class', 'HerReplayBuffer'), ('replay_buffer_kwargs', "dict( goal_selection_strategy='future', n_sampled_goal=4 )"), ('normalize_kwargs',{'norm_obs':False,'norm_reward':False})]),
    "ppo": OrderedDict([('batch_size', 256), ('ent_coef', 0.01), ('gae_lambda', 0.9), ('gamma', 0.99), ('learning_rate', 'lin_0.0001'), ('max_grad_norm', 0.5), ('n_envs', 8), ('n_epochs', 20), ('n_steps', 8), ('n_timesteps', 25000.0), ('normalize_advantage', False), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(log_std_init=-2, ortho_init=False)'), ('use_sde', True), ('vf_coef', 0.4), ('normalize', False), ('normalize_kwargs', {'norm_obs': False, 'norm_reward': False})]),
}

ALGOS = {
    "sac": SAC,
    "ppo": PPO,
}


class StableBaseLineTrainedAgent(RLAgent):
    def __init__(
            self,
            episodes: int,
            decaying_eps: bool,
            epsilon: float,
            learning_rate: float,
            gamma: float,
            problem_name: str,
            env_name: str,
            exp_id: int,
            folder: str,
            algo: str,
            load_best: bool,
            load_checkpoint: bool,
            load_last_checkpoint: bool,
            goal_hypothesis: Optional[str] = None,
    ):
        super().__init__(
            episodes=episodes,
            decaying_eps=decaying_eps,
            epsilon=epsilon,
            learning_rate=learning_rate,
            gamma=gamma,
            env_name=env_name,
            problem_name=problem_name,
            goal_hypothesis=goal_hypothesis
        )
        self._model = self.algorithm.load(self._model_file_path)

        self.env = env

        self._goal = None
        self.set_goal()

        self.test()

    @property
    def goal(self):
        return np.array([self._goal], dtype=np.float32)

    def set_goal(self):
        raise NotImplementedError()

    @property
    def actor(self):
        raise NotImplementedError()

    def get_mean_and_std_dev(self, observation):
        raise NotImplementedError()

    def learn(self):
        print("HuggingFaceTrainedAgent learning func!")

    @staticmethod
    def _validate_is_info_success(infos) -> bool:
        is_success_list = [info_dict["is_success"] for info_dict in infos if "is_success" in info_dict]
        assert len(is_success_list) == 1, "is_success array len is not 1, info:{infos}"

        return is_success_list[0]

    def test(self):
        obs = self.env.reset()
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        deterministic = True
        episode_reward = 0.0
        ep_len = 0
        generator = range(5000)
        for i in generator:
            # print(f"iteration {i}:{obs=}")
            action, lstm_states = self.model.predict(
                obs,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, reward, done, infos = self.env.step(action)

            assert len(reward) == 1, f"length of rewards list is not 1, rewards:{reward}"
            is_success = self._validate_is_info_success(infos)
            # print(f"(action,is_done,info):({action},{done},{infos})")
            if is_success:
                print(f"breaking due to GG, took {i} steps")
                break
            episode_start = done

            episode_reward += reward[0]
            ep_len += 1
        self.env.close()


class SACTrainedAgent(StableBaseLineTrainedAgent):
    NAME = "sac"

    def __init__(self,
                 problem_name: str,
                 env_name: str,
                 episodes: int = -1,
                 decaying_eps: bool = False,
                 epsilon: float = 0.95,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 exp_id: int = 0,
                 folder: str = "dataset/{env_name}/models",
                 load_best: bool = 0,
                 load_checkpoint: bool = None,
                 load_last_checkpoint: bool = False):
        print(f"folder:{folder.format(env_name=env_name)}")
        super().__init__(episodes=episodes,
                         decaying_eps=decaying_eps,
                         epsilon=epsilon,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         problem_name=problem_name,
                         env_name=env_name,
                         exp_id=exp_id,
                         folder=folder.format(env_name=env_name),
                         algo=SACTrainedAgent.NAME,
                         load_best=load_best,
                         load_checkpoint=load_checkpoint,
                         load_last_checkpoint=load_last_checkpoint)

    @property
    def actor(self):
        return self.model.actor

    def set_goal(self):
        self._goal = extract_goal(env_name=self.env_name, env=self.env.envs[0])
        print(f"{self._goal=}")

    def get_mean_and_std_dev(self, observation):
        tensor_observation, _ = self.model.actor.obs_to_tensor(observation)

        mean_actions, log_std_dev, kwargs = self.model.actor.get_action_dist_params(tensor_observation)
        probability_dist = self.model.actor.action_dist.proba_distribution(
            mean_actions=mean_actions,
            log_std=log_std_dev
        )
        actor_means = probability_dist.get_actions(True).cpu().detach().numpy()
        log_std_dev = log_std_dev.cpu().detach().numpy()

        return actor_means, log_std_dev


class PPOTrainedAgent(StableBaseLineTrainedAgent):
    NAME = "ppo"

    def __init__(self,
                 problem_name: str,
                 env_name: str,
                 models_dir: str,
                 episodes: int = -1,
                 decaying_eps: bool = False,
                 epsilon: float = 0.95,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 exp_id: int = 0,
                 folder: str = "dataset/{env_name}/models",
                 load_best: bool = 0,
                 load_checkpoint: bool = None,
                 load_last_checkpoint: bool = False,
                 goal_hypothesis: Optional[str] = None,
    ):
        print(f"folder:{folder.format(env_name=env_name)}")
        super().__init__(episodes=episodes,
                         decaying_eps=decaying_eps,
                         epsilon=epsilon,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         problem_name=problem_name,
                         env_name=env_name,
                         models_dir=models_dir,
                         exp_id=exp_id,
                         folder=folder.format(env_name=env_name),
                         algo=PPOTrainedAgent.NAME,
                         load_best=load_best,
                         load_checkpoint=load_checkpoint,
                         load_last_checkpoint=load_last_checkpoint,
                         goal_hypothesis=goal_hypothesis
        )

    @property
    def actor(self):
        return self.model.policy

    def get_mean_and_std_dev(self, observation):
        self.actor.set_training_mode(False)

        tensor_observation, _ = self.actor.obs_to_tensor(observation)
        distribution = self.actor.get_distribution(tensor_observation)

        actor_means = distribution.distribution.mean.cpu().detach().numpy()
        log_std_dev = distribution.distribution.stddev.cpu().detach().numpy()
        if isinstance(self.actor.action_space, gymnasium.spaces.Box):
            actor_means = np.clip(
                actor_means,
                self.actor.action_space.low,
                self.actor.action_space.high
            )
        return actor_means, log_std_dev

    def set_goal(self):
        self._goal = extract_goal(env_name=self.env_name, env=self.env.envs[0])
        print(f"{self._goal=}")
