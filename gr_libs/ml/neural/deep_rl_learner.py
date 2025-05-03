from collections import OrderedDict
import gc
from types import MethodType
from typing import List, Tuple
import numpy as np
import cv2

from gr_libs.environment.environment import EnvProperty

if __name__ != "__main__":
    from gr_libs.ml.utils.storage import get_agent_model_dir
    from gr_libs.ml.utils.format import random_subset_with_order
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from gr_libs.ml.utils import device
import gymnasium as gym

# built-in python modules
import random
import os
import sys

# TODO do we need this?
NETWORK_SETUP = {
    SAC: OrderedDict(
        [
            ("batch_size", 512),
            ("buffer_size", 100000),
            ("ent_coef", "auto"),
            ("gamma", 0.95),
            ("learning_rate", 0.001),
            ("learning_starts", 5000),
            ("n_timesteps", 50000.0),
            ("normalize", "{'norm_obs': False, 'norm_reward': False}"),
            ("policy", "MultiInputPolicy"),
            ("policy_kwargs", "dict(net_arch=[64, 64])"),
            ("replay_buffer_class", "HerReplayBuffer"),
            (
                "replay_buffer_kwargs",
                "dict( goal_selection_strategy='future', n_sampled_goal=4 )",
            ),
            ("normalize_kwargs", {"norm_obs": False, "norm_reward": False}),
        ]
    ),
    # "tqc": OrderedDict([('batch_size', 256), ('buffer_size', 1000000), ('ent_coef', 'auto'), ('env_wrapper', ['sb3_contrib.common.wrappers.TimeFeatureWrapper']), ('gamma', 0.95), ('learning_rate', 0.001), ('learning_starts', 1000), ('n_timesteps', 25000.0), ('normalize', False), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(net_arch=[64, 64])'), ('replay_buffer_class', 'HerReplayBuffer'), ('replay_buffer_kwargs', "dict( goal_selection_strategy='future', n_sampled_goal=4 )"), ('normalize_kwargs',{'norm_obs':False,'norm_reward':False})]),
    PPO: OrderedDict(
        [
            ("batch_size", 256),
            ("ent_coef", 0.01),
            ("gae_lambda", 0.9),
            ("gamma", 0.99),
            ("learning_rate", "lin_0.0001"),
            ("max_grad_norm", 0.5),
            ("n_envs", 8),
            ("n_epochs", 20),
            ("n_steps", 8),
            ("n_timesteps", 25000.0),
            ("normalize_advantage", False),
            ("policy", "MultiInputPolicy"),
            ("policy_kwargs", "dict(log_std_init=-2, ortho_init=False)"),
            ("use_sde", True),
            ("vf_coef", 0.4),
            ("normalize", False),
            ("normalize_kwargs", {"norm_obs": False, "norm_reward": False}),
        ]
    ),
}


class DeepRLAgent:
    def __init__(
        self,
        domain_name: str,
        problem_name: str,
        num_timesteps: float,
        env_prop: EnvProperty,
        algorithm: BaseAlgorithm = SAC,
        reward_threshold: float = 450,
        exploration_rate=None,
    ):
        # Need to change reward threshold to change according to which task the agent is training on, becuase it changes from task to task.
        env_kwargs = {"id": problem_name, "render_mode": "rgb_array"}
        assert algorithm in [SAC, PPO, TD3]

        self.domain_name = domain_name
        self.problem_name = problem_name
        self.env_prop = env_prop
        self.exploration_rate = exploration_rate

        self._model_directory = get_agent_model_dir(
            domain_name=self.domain_name,
            model_name=problem_name,
            class_name=algorithm.__name__,
        )
        self.env = self.env_prop.create_vec_env(env_kwargs)
        self._actions_space = self.env.action_space

        # first_support: SB3 models from RL zoo, with the .zip format.
        if os.path.exists(os.path.join(self._model_directory, "saved_model.zip")):
            # TODO check if it's ncessary to give these to the model.load if loading from rl zoo
            self._model_file_path = os.path.join(
                self._model_directory, "saved_model.zip"
            )
            self.model_kwargs = {
                "custom_objects": {
                    "learning_rate": 0.0,
                    "lr_schedule": lambda _: 0.0,
                    "clip_range": lambda _: 0.0,
                },
                "seed": 0,
                "buffer_size": 1,
            }
        # second support: models saved with SB3's model.save, which is saved as a formatted .pth file.
        else:
            self.model_kwargs = {}
            self._model_file_path = os.path.join(
                self._model_directory, "saved_model.pth"
            )

        self.algorithm = algorithm
        self.reward_threshold = reward_threshold
        self.num_timesteps = num_timesteps

    def save_model(self):
        self._model.save(self._model_file_path)

    def try_recording_video(self, video_path, desired=None):
        num_tries = 0
        while True:
            if num_tries >= 10:
                assert False, "agent keeps failing on recording an optimal obs."
            try:
                self.record_video(video_path, desired)
                break
            except Exception as e:
                num_tries += 1
        # print(f"sequence to {self.problem_name} is:\n\t{steps}\ngenerating image at {img_path}.")
        print(f"generated sequence video at {video_path}.")

    def record_video(self, video_path, desired=None):
        """Record a video of the agent's performance."""
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        fps = 30.0
        # if is_gc:
        # 	assert goal_idx != None
        # 	self.reset_with_goal_idx(goal_idx)
        # else:
        # 	assert goal_idx == None
        self.env.reset()
        frame_size = (
            self.env.render(mode="rgb_array").shape[1],
            self.env.render(mode="rgb_array").shape[0],
        )
        video_path = os.path.join(video_path, "plan_video.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        general_done, success_done = False, False
        gc.collect()
        obs = self.env.reset()
        self.env_prop.change_goal_to_specific_desired(obs, desired)
        counter = 0
        while not (general_done or success_done):
            counter += 1
            action, _states = self._model.predict(obs, deterministic=False)
            obs, rewards, general_done, info = self.env.step(action)
            if isinstance(general_done, np.ndarray):
                general_done = general_done[0]
            self.env_prop.change_goal_to_specific_desired(obs, desired)
            if "success" in info[0].keys():
                success_done = info[0][
                    "success"
                ]  # make sure the agent actually reached the goal within the max time
            elif "is_success" in info[0].keys():
                success_done = info[0][
                    "is_success"
                ]  # make sure the agent actually reached the goal within the max time
            elif "step_task_completions" in info[0].keys():
                success_done = (
                    len(info[0]["step_task_completions"]) == 1
                )  # bug of dummyVecEnv, it removes the episode_task_completions from the info dict.
            else:
                raise NotImplementedError(
                    "no other option for any of the environments."
                )
            frame = self.env.render()
            success_done = self.env_prop.change_done_by_specific_desired(
                obs, desired, success_done
            )
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if general_done == False != success_done == True:
            assert (
                desired is not None
            ), f"general_done is false but success_done is true, and desired is None. This should never happen, since the \
					 							 environment will say 'done' is false (general_done) while the observation will be close to the goal (success_done) \
												 only in case we incorporated a 'desired' when generating the observation."
        elif general_done == True != success_done == False:
            raise Exception("general_done is true but success_done is false")
        self.env.close()
        video_writer.release()

    def load_model(self):
        self._model = self.algorithm.load(
            self._model_file_path, env=self.env, device=device, **self.model_kwargs
        )

    def learn(self):
        if os.path.exists(self._model_file_path):
            print(f"Loading pre-existing model in {self._model_file_path}")
            self.load_model()
        else:
            # Stop training when the model reaches the reward threshold
            # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=self.reward_threshold, verbose=1)
            # eval_callback = EvalCallback(self.env, best_model_save_path="./logs/",
            #                  log_path="./logs/", eval_freq=500, callback_on_new_best=callback_on_best, verbose=1, render=True)
            # self._model.learn(total_timesteps=self.num_timesteps, progress_bar=True, callback=eval_callback)
            print(f"No existing model in {self._model_file_path}, starting learning")
            if self.exploration_rate != None:
                self._model = self.algorithm(
                    "MultiInputPolicy",
                    self.env,
                    ent_coef=self.exploration_rate,
                    verbose=1,
                )
            else:
                self._model = self.algorithm("MultiInputPolicy", self.env, verbose=1)
            self._model.learn(
                total_timesteps=self.num_timesteps, progress_bar=True
            )  # comment this in a normal env
            self.save_model()

    def safe_env_reset(self):
        try:
            obs = self.env.reset()
        except Exception as e:
            kwargs = {"id": self.problem_name, "render_mode": "rgb_array"}
            self.env = self.env_prop.create_vec_env(kwargs)
            obs = self.env.reset()
        return obs

    def get_mean_and_std_dev(self, observation):
        if self.algorithm == SAC:
            tensor_observation, _ = self._model.actor.obs_to_tensor(observation)

            mean_actions, log_std_dev, kwargs = (
                self._model.actor.get_action_dist_params(tensor_observation)
            )
            probability_dist = self._model.actor.action_dist.proba_distribution(
                mean_actions=mean_actions, log_std=log_std_dev
            )
            actor_means = probability_dist.get_actions(True).cpu().detach().numpy()
            log_std_dev = log_std_dev.cpu().detach().numpy()
        elif self.algorithm == PPO:
            self._model.policy.set_training_mode(False)
            tensor_observation, _ = self._model.policy.obs_to_tensor(observation)
            distribution = self._model.policy.get_distribution(tensor_observation)

            actor_means = distribution.distribution.mean.cpu().detach().numpy()
            log_std_dev = distribution.distribution.stddev.cpu().detach().numpy()
            if isinstance(self._model.policy.action_space, gym.spaces.Box):
                actor_means = np.clip(
                    actor_means,
                    self._model.policy.action_space.low,
                    self._model.policy.action_space.high,
                )
            return actor_means, log_std_dev
        else:
            assert False
        return actor_means, log_std_dev

    # fits agents that generated observations in the form of: list of tuples, each tuple a single step\frame with size 2, comprised of obs and action.
    # the function squashes the 2d array of obs and action in a 1d array, concatenating their values together for training.
    def simplify_observation(self, observation):
        return [
            np.concatenate(
                (
                    np.array(obs).reshape(obs.shape[-1]),
                    np.array(action[0]).reshape(action[0].shape[-1]),
                )
            )
            for (obs, action) in observation
        ]

    def add_random_optimalism(self, observations, action, constant_initial_action):
        if len(observations) > 3:
            for i in range(0, len(action[0])):
                action[0][i] += random.uniform(
                    -0.01 * action[0][i], 0.01 * action[0][i]
                )
        else:  # just walk in a specific random direction to enable diverse plans
            action = np.array(np.array([constant_initial_action]), None)

    def generate_partial_observation(
        self,
        action_selection_method,
        percentage,
        is_consecutive,
        save_fig=False,
        fig_path=None,
        random_optimalism=True,
    ):
        steps = self.generate_observation(
            action_selection_method,
            save_fig=save_fig,
            random_optimalism=random_optimalism,
            fig_path=fig_path,
        )  # steps are a full observation
        return random_subset_with_order(
            steps, (int)(percentage * len(steps)), is_consecutive
        )

    def generate_observation(
        self,
        action_selection_method: MethodType,
        random_optimalism,
        save_fig=False,
        fig_path=None,
        with_dict=False,
        desired=None,
    ) -> List[
        Tuple[np.ndarray, np.ndarray]
    ]:  # TODO make sure to add a linter to alert when a method doesn't accept or return the type it should
        if save_fig == False:
            assert (
                fig_path == None
            ), "You can't specify a vid path when you don't even save the figure."
        else:
            assert (
                fig_path != None
            ), "You need to specify a vid path when you save the figure."
        # The try-except is a bug fix for the env not being reset properly in panda. If someone wants to check why and provide a robust solution they're welcome.
        obs = self.safe_env_reset()
        self.env_prop.change_goal_to_specific_desired(obs, desired)
        observations = []
        is_successful_observation_made = False
        num_of_insuccessful_attempts = 0
        while not is_successful_observation_made:
            is_successful_observation_made = True  # start as true, if this isn't the case (crash/death/truncation instead of success)
            if random_optimalism:
                constant_initial_action = self.env.action_space.sample()
            while True:
                from gr_libs.metrics.metrics import stochastic_amplified_selection

                deterministic = (
                    action_selection_method != stochastic_amplified_selection
                )
                action, _states = self._model.predict(obs, deterministic=deterministic)
                if (
                    random_optimalism
                ):  # get the right direction and then start inserting noise to still get a relatively optimal plan
                    self.add_random_optimalism(obs, action, constant_initial_action)
                if with_dict:
                    observations.append((obs, action))
                else:
                    observations.append((obs["observation"], action))
                obs, reward, done, info = self.env.step(action)
                self.env_prop.change_goal_to_specific_desired(obs, desired)
                general_done = self.env_prop.is_done(done)
                success_done = self.env_prop.is_success(info)
                success_done = self.env_prop.change_done_by_specific_desired(
                    obs, desired, success_done
                )
                if general_done == True and success_done == False:
                    # it could be that the stochasticity inserted into the actions made the agent die/crash. we don't want this observation: it's an insuccessful attempt.
                    num_of_insuccessful_attempts += 1
                    # print(f"for agent for problem {self.problem_name}, its done {len(observations)} steps, and got to a situation where general_done != success_done, for the {num_of_insuccessful_attempts} time.")
                    if num_of_insuccessful_attempts > 50:
                        # print(f"got more then 10 insuccessful attempts!")
                        assert (
                            general_done == success_done
                        ), f"failed on goal: {obs['desired']}"  # we want to make sure the episode is done only when the agent has actually succeeded with the task.
                    else:
                        # try again by breaking inner loop. everything is set up to be like the beginning of the function.
                        is_successful_observation_made = False
                        obs = self.safe_env_reset()
                        self.env_prop.change_goal_to_specific_desired(obs, desired)
                        observations = (
                            []
                        )  # we want to re-accumulate the observations from scratch, have another try
                        break
                elif general_done == False and success_done == False:
                    continue
                elif general_done == True and success_done == True:
                    if num_of_insuccessful_attempts > 0:
                        pass  # print(f"after {num_of_insuccessful_attempts}, finally I succeeded!")
                    break
                elif general_done == False and success_done == True:
                    # The environment will say 'done' is false (general_done) while the observation will be close to the goal (success_done)
                    # only in case we incorporated a 'desired' when generating the observation.
                    assert (
                        desired is not None
                    ), f"general_done is false but success_done is true, and desired is None. This should never happen, since the \
					 							 environment will say 'done' is false (general_done) while the observation will be close to the goal (success_done) \
												 only in case we incorporated a 'desired' when generating the observation."
                    break

        if save_fig:
            self.try_recording_video(fig_path, desired)

        self.env.close()
        return observations


class GCDeepRLAgent(DeepRLAgent):
    def generate_partial_observation(
        self,
        action_selection_method,
        percentage,
        is_consecutive,
        goal_directed_problem=None,
        goal_directed_goal=None,
        save_fig=False,
        fig_path=None,
        random_optimalism=True,
    ):
        steps = self.generate_observation(
            action_selection_method,
            save_fig=save_fig,
            fig_path=fig_path,
            random_optimalism=random_optimalism,
            goal_directed_problem=goal_directed_problem,
            goal_directed_goal=goal_directed_goal,
        )  # steps are a full observation
        return random_subset_with_order(
            steps, (int)(percentage * len(steps)), is_consecutive
        )

    # TODO move the goal_directed_goal and/or goal_directed_problem mechanism to be a property of the env_property, so deep_rl_learner doesn't depend on it and holds this logic so heavily.
    # Generate observation with goal_directed_goal or goal_directed_problem is only possible for a GC agent, otherwise - the agent can't act optimally to that new goal.
    def generate_observation(
        self,
        action_selection_method: MethodType,
        random_optimalism,
        goal_directed_problem=None,
        goal_directed_goal=None,
        save_fig=False,
        fig_path=None,
        with_dict=False,
    ):
        if save_fig:
            assert (
                fig_path != None
            ), "You need to specify a vid path when you save the figure."
        else:
            assert fig_path == None
        # goal_directed_problem employs the GC agent in a new env with a static, predefined goal, and has him generate an observation sequence in it.
        if goal_directed_problem:
            assert (
                goal_directed_goal == None
            ), "can't give goal directed goal and also goal directed problem for the sake of sequence generation by a general agent"
            kwargs = {"id": goal_directed_problem, "render_mode": "rgb_array"}
            self.env = self.env_prop.create_vec_env(kwargs)
            orig_env = self.env
            observations = super().generate_observation(
                action_selection_method=action_selection_method,
                random_optimalism=random_optimalism,
                save_fig=save_fig,
                fig_path=fig_path,
                with_dict=with_dict,
            )
            self.env = orig_env
        # goal_directed_goal employs the agent in the same env on which it trained - with goals that change with every episode sampled from the goal space.
        # but we manually change the 'desired' part of the observation to be the goal_directed_goal and edit the id_success and is_done accordingly.
        else:
            assert (
                goal_directed_problem == None
            ), "can't give goal directed goal and also goal directed problem for the sake of sequence generation by a general agent"
            observations = super().generate_observation(
                action_selection_method=action_selection_method,
                random_optimalism=random_optimalism,
                save_fig=save_fig,
                fig_path=fig_path,
                with_dict=with_dict,
                desired=goal_directed_goal,
            )  # TODO tutorial on how to use the deepRLAgent for sequence generation and examination and plotting of the sequence
        return observations
