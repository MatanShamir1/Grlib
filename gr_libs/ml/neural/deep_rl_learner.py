from collections import OrderedDict
import gc
from types import MethodType
from typing import List, Tuple
import gymnasium as gym
import numpy as np
import cv2

HACK_HAPPENED = False

if __name__ != "__main__":
	from gr_libs.ml.utils.storage import get_agent_model_dir
	from gr_libs.ml.utils.format import random_subset_with_order
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gr_libs.ml.utils import device

# built-in python modules
import random
import os
import sys

def create_vec_env(kwargs):
	# create the model, it will not be a pretrained one anyway
	# env = gym.make(**kwargs)
	env = gym.make(**kwargs)
	return DummyVecEnv([lambda: env])

def change_goal_to_specific_desired(obs, desired):
	if desired is not None:
		obs['desired_goal'] = desired
	# try:
	# 	if desired!=None: obs['desired_goal'] = desired
	# except Exception as e:
	# 	try:
	# 		if all(desired!=None): obs['desired_goal'] = desired
	# 	except Exception as e:
	# 		if all([desiredy!=None for desiredish in desired for desiredy in desiredish]): obs['desired_goal'] = desired


NETWORK_SETUP = {
	SAC: OrderedDict([('batch_size', 512), ('buffer_size', 100000), ('ent_coef', 'auto'), ('gamma', 0.95), ('learning_rate', 0.001), ('learning_starts', 5000), ('n_timesteps', 50000.0), ('normalize', "{'norm_obs': False, 'norm_reward': False}"), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(net_arch=[64, 64])'), ('replay_buffer_class', 'HerReplayBuffer'), ('replay_buffer_kwargs', "dict( goal_selection_strategy='future', n_sampled_goal=4 )"), ('normalize_kwargs', {'norm_obs': False, 'norm_reward': False})]),
	#"tqc": OrderedDict([('batch_size', 256), ('buffer_size', 1000000), ('ent_coef', 'auto'), ('env_wrapper', ['sb3_contrib.common.wrappers.TimeFeatureWrapper']), ('gamma', 0.95), ('learning_rate', 0.001), ('learning_starts', 1000), ('n_timesteps', 25000.0), ('normalize', False), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(net_arch=[64, 64])'), ('replay_buffer_class', 'HerReplayBuffer'), ('replay_buffer_kwargs', "dict( goal_selection_strategy='future', n_sampled_goal=4 )"), ('normalize_kwargs',{'norm_obs':False,'norm_reward':False})]),
	PPO: OrderedDict([('batch_size', 256), ('ent_coef', 0.01), ('gae_lambda', 0.9), ('gamma', 0.99), ('learning_rate', 'lin_0.0001'), ('max_grad_norm', 0.5), ('n_envs', 8), ('n_epochs', 20), ('n_steps', 8), ('n_timesteps', 25000.0), ('normalize_advantage', False), ('policy', 'MultiInputPolicy'), ('policy_kwargs', 'dict(log_std_init=-2, ortho_init=False)'), ('use_sde', True), ('vf_coef', 0.4), ('normalize', False), ('normalize_kwargs', {'norm_obs': False, 'norm_reward': False})]),
}

class DeepRLAgent():
	def __init__(self, domain_name: str, problem_name: str, num_timesteps:float, algorithm=SAC, reward_threshold: float=450,
			  	 exploration_rate=None):
		# Need to change reward threshold to change according to which task the agent is training on, becuase it changes from task to task.
		kwargs = {"id":problem_name, "render_mode":"rgb_array"}
		
		self.domain_name = domain_name
		self.problem_name = problem_name

		self._model_directory = get_agent_model_dir(domain_name=self.domain_name, model_name=problem_name, class_name=algorithm.__name__)
		if os.path.exists(os.path.join(self._model_directory, "saved_model.zip")):
			self.pre_trained_model = True
			self._model_file_path = os.path.join(self._model_directory, "saved_model.zip")
		else:
			self.pre_trained_model = False
			self.env = create_vec_env(kwargs)
			self._actions_space = self.env.action_space
			if exploration_rate != None: self._model = algorithm("MultiInputPolicy", self.env, ent_coef=exploration_rate, verbose=1)
			else: self._model = algorithm("MultiInputPolicy", self.env, verbose=1)
			self._model_file_path = os.path.join(self._model_directory, "saved_model.pth")
		self.algorithm = algorithm
		self.reward_threshold = reward_threshold
		self.num_timesteps = num_timesteps

	def save_model(self):
		self._model.save(self._model_file_path)

	def record_video(self, video_path, desired=None):
		global HACK_HAPPENED
		"""Record a video of the agent's performance."""
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		fps = 30.0
		# if is_gc:
		# 	assert goal_idx != None
		# 	self.reset_with_goal_idx(goal_idx)
		# else:
		# 	assert goal_idx == None
		self.env.reset()
		frame_size = (self.env.render(mode='rgb_array').shape[1], self.env.render(mode='rgb_array').shape[0])
		video_path = os.path.join(video_path, "plan_video.mp4")
		video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
		general_done, success_done = False, False
		gc.collect()
		obs = self.env.reset()
		change_goal_to_specific_desired(obs, desired)
		counter = 0
		while not (general_done or success_done):
			counter += 1
			action, _states = self._model.predict(obs, deterministic=False)
			obs, rewards, general_done, info = self.env.step(action)
			if isinstance(general_done, np.ndarray): general_done = general_done[0]
			change_goal_to_specific_desired(obs, desired)
			if "success" in info[0].keys(): success_done = info[0]["success"] # make sure the agent actually reached the goal within the max time
			elif "is_success" in info[0].keys(): success_done = info[0]["is_success"] # make sure the agent actually reached the goal within the max time
			elif "step_task_completions" in info[0].keys(): success_done = (len(info[0]["step_task_completions"]) == 1) # bug of dummyVecEnv, it removes the episode_task_completions from the info dict.
			else: raise NotImplementedError("no other option for any of the environments.")
			frame = self.env.render()
			success_done = self.change_done_by_specific_desired(obs, desired, success_done)
			video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
		if general_done == False != success_done == True:
			assert HACK_HAPPENED
		elif general_done == True != success_done == False:
			raise Exception("general_done is true but success_done is false")
		self.env.close()
		video_writer.release()

	#def set_success_done(self, success_done, desired, )

	def change_done_by_specific_desired(self, obs, desired, old_success_done):
		global HACK_HAPPENED
		try:
			if desired!=None:
				HACK_HAPPENED = True
				if 'Panda' in self.problem_name:
					assert obs['achieved_goal'].shape == desired.shape
					d = np.linalg.norm(obs['achieved_goal'] - desired, axis=-1)
					# print(f"achieved_goal:{achieved_goal}, desired_goal:{desired_goal}, distance:{d}, is finished:{d < self.distance_threshold}")
					return (d < 0.04)[0]
				elif 'Parking' in self.problem_name: # shuoldn't be used for now
					# TODO
					return self.env.task.is_success()
			else:
				return old_success_done
		except Exception as e:
			try:
				if all(desired!=None):
					HACK_HAPPENED = True
					if 'Panda' in self.problem_name:
						assert obs['achieved_goal'].shape == desired.shape
						d = np.linalg.norm(obs['achieved_goal'] - desired, axis=-1)
						# print(f"achieved_goal:{achieved_goal}, desired_goal:{desired_goal}, distance:{d}, is finished:{d < self.distance_threshold}")
						return (d < 0.04)[0]
					elif 'Parking' in self.problem_name:
						# TODO add all of this to the environment property. recognizer shouldn't know anything about it.
						return self.env.task.is_success()
				else:
					return old_success_done
			except Exception as e:
				if all([desiredy!=None for desiredish in desired for desiredy in desiredish]):
					HACK_HAPPENED = True
					if 'Panda' in self.problem_name:
						assert obs['achieved_goal'].shape == desired.shape
						d = np.linalg.norm(obs['achieved_goal'] - desired, axis=-1)
						# print(f"achieved_goal:{achieved_goal}, desired_goal:{desired_goal}, distance:{d}, is finished:{d < self.distance_threshold}")
						return (d < 0.04)[0]
					elif 'Parking' in self.problem_name:
						# TODO
						return self.env.task.is_success()
				else:
					return old_success_done

	def load_model(self):
		self._model = self.algorithm.load(self._model_file_path, env=self.env, device=device)

	def learn(self):
		if os.path.exists(self._model_file_path):
			print(f"Loading pre-existing model in {self._model_file_path}")
			if self.pre_trained_model:
				def test(env):
					obs = env.reset()
					lstm_states = None
					episode_start = np.ones((1,), dtype=bool)
					deterministic = True
					episode_reward = 0.0
					ep_len = 0
					generator = range(5000)
					for i in generator:
						# print(f"iteration {i}:{obs=}")
						action, lstm_states = self._model.predict(
							obs,  # type: ignore[arg-type]
							state=lstm_states,
							episode_start=episode_start,
							deterministic=deterministic,
						)
						obs, reward, done, infos = env.step(action)

						assert len(reward) == 1, f"length of rewards list is not 1, rewards:{reward}"
						if "success" in infos[0].keys(): is_success = infos[0]["success"] # make sure the agent actually reached the goal within the max time
						elif "is_success" in infos[0].keys(): is_success = infos[0]["is_success"] # make sure the agent actually reached the goal within the max time
						elif "step_task_completions" in infos[0].keys(): is_success = (len(infos[0]["step_task_completions"]) == 1) # bug of dummyVecEnv, it removes the episode_task_completions from the info dict.
						else: raise NotImplementedError("no other option for any of the environments.")
						# print(f"(action,is_done,info):({action},{done},{infos})")
						if is_success:
							#print(f"breaking due to GG, took {i} steps")
							break
						episode_start = done

						episode_reward += reward[0]
						ep_len += 1
					env.close()
				custom_objects = {
					"learning_rate": 0.0,
					"lr_schedule": lambda _: 0.0,
					"clip_range": lambda _: 0.0,
				}
				kwargs = {"id": self.problem_name, "render_mode": "rgb_array"}
				self.env = create_vec_env(kwargs)
				self._actions_space = self.env.action_space
				kwargs = {'seed': 0, 'buffer_size': 1}

				self._model = self.algorithm.load(self._model_file_path, env=self.env, custom_objects=custom_objects, device=device, **kwargs)
				test(self.env)
			else:
				self.load_model()
		else:
			# Stop training when the model reaches the reward threshold
			# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=self.reward_threshold, verbose=1)
			# eval_callback = EvalCallback(self.env, best_model_save_path="./logs/",
			#                  log_path="./logs/", eval_freq=500, callback_on_new_best=callback_on_best, verbose=1, render=True)
			# self._model.learn(total_timesteps=self.num_timesteps, progress_bar=True, callback=eval_callback)
			print(f"No existing model in {self._model_file_path}, starting learning")
			self._model.learn(total_timesteps=self.num_timesteps, progress_bar=True) # comment this in a normal env
			self.save_model()

	def get_mean_and_std_dev(self, observation):
		if self.algorithm == SAC:
			tensor_observation, _ = self._model.actor.obs_to_tensor(observation)

			mean_actions, log_std_dev, kwargs = self._model.actor.get_action_dist_params(tensor_observation)
			probability_dist = self._model.actor.action_dist.proba_distribution(
				mean_actions=mean_actions,
				log_std=log_std_dev
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
					self._model.policy.action_space.high
				)
			return actor_means, log_std_dev
		else:
			assert False
		return actor_means, log_std_dev

	# fits agents that generated observations in the form of: list of tuples, each tuple a single step\frame with size 2, comprised of obs and action.
	# the function squashes the 2d array of obs and action in a 1d array, concatenating their values together for training.
	def simplify_observation(self, observation):
		return [np.concatenate((np.array(obs).reshape(obs.shape[-1]),np.array(action[0]).reshape(action[0].shape[-1]))) for (obs,action) in observation]

	def generate_partial_observation(self, action_selection_method, percentage, is_consecutive, save_fig=False, fig_path=None, random_optimalism=True):
		steps = self.generate_observation(action_selection_method, save_fig=save_fig, random_optimalism=random_optimalism, fig_path=fig_path) # steps are a full observation
		return random_subset_with_order(steps, (int)(percentage * len(steps)), is_consecutive)

	def generate_observation(self, action_selection_method: MethodType, random_optimalism, save_fig=False, env_prop=None,
							 fig_path=None, with_dict=False, desired=None) -> List[Tuple[np.ndarray, np.ndarray]]: # TODO make sure to add a linter to alert when a method doesn't accept or return the type it should
		if save_fig == False:
			assert fig_path == None, "You can't specify a vid path when you don't even save the figure."
		else:
			assert fig_path != None, "You need to specify a vid path when you save the figure."
		# The try-except is a bug fix for the env not being reset properly in panda. If someone wants to check why and provide a robust solution they're welcome.
		try:
			obs = self.env.reset()
			change_goal_to_specific_desired(obs, desired)
		except Exception as e:
			kwargs = {"id": self.problem_name, "render_mode": "rgb_array"}
			self.env = create_vec_env(kwargs)
			obs = self.env.reset()
			change_goal_to_specific_desired(obs, desired)
		observations = []
		is_successful_observation_made = False
		num_of_insuccessful_attempts = 0
		while not is_successful_observation_made:
			is_successful_observation_made = True # start as true, if this isn't the case (crash/death/truncation instead of success)
			if random_optimalism:
				constant_initial_action = self.env.action_space.sample()
			while True:
				from gr_libs.metrics.metrics import stochastic_amplified_selection
				deterministic = action_selection_method != stochastic_amplified_selection
				action, _states = self._model.predict(obs, deterministic=deterministic)
				if random_optimalism: # get the right direction and then start inserting noise to still get a relatively optimal plan
					if len(observations) > 3:
						for i in range(0, len(action[0])):
							action[0][i] += random.uniform(-0.01 * action[0][i], 0.01 * action[0][i])
					else: # just walk in a specific random direction to enable diverse plans
						action = np.array(np.array([constant_initial_action]), None)
				if with_dict: observations.append((obs, action))
				else: observations.append((obs['observation'], action))
				obs, reward, done, info = self.env.step(action)
				change_goal_to_specific_desired(obs, desired)
				if isinstance(done, np.ndarray): general_done = done[0]
				else: general_done = done
				if "success" in info[0].keys(): success_done = info[0]["success"]
				elif "is_success" in info[0].keys(): success_done = info[0]["is_success"]
				elif "step_task_completions" in info[0].keys(): success_done = info[0]["step_task_completions"]
				else: raise NotImplementedError("no other option for any of the environments.")
				success_done = self.change_done_by_specific_desired(obs, desired, success_done)
				if general_done == True and success_done == False:
					# it could be that the stochasticity inserted into the actions made the agent die/crash. we don't want this observation.
					num_of_insuccessful_attempts += 1
					# print(f"for agent for problem {self.problem_name}, its done {len(observations)} steps, and got to a situation where general_done != success_done, for the {num_of_insuccessful_attempts} time.")
					if num_of_insuccessful_attempts > 50:
						# print(f"got more then 10 insuccessful attempts. fuak!")
						assert general_done == success_done, f"failed on goal: {obs['desired']}" # we want to make sure the episode is done only when the agent has actually succeeded with the task.
					else:
						# try again by breaking inner loop. everything is set up to be like the beginning of the function.
						is_successful_observation_made = False
						try:
							obs = self.env.reset()
							change_goal_to_specific_desired(obs, desired)
						except Exception as e:
							kwargs = {"id": self.problem_name, "render_mode": "rgb_array"}
							self.env = create_vec_env(kwargs)
							obs = self.env.reset()
							change_goal_to_specific_desired(obs, desired)
						observations = [] # we want to re-accumulate the observations from scratch, have another try
						break
				elif general_done == False and success_done == False:
					continue
				elif general_done == True and success_done == True:
					if num_of_insuccessful_attempts > 0:
						pass # print(f"after {num_of_insuccessful_attempts}, finally I succeeded!")
					break
				elif general_done == False and success_done == True:
					assert HACK_HAPPENED == True # happens only if hack happened
					break
		# self.env.close()
		if save_fig:
			num_tries = 0
			while True:
				if num_tries >= 10:
					assert False, "agent keeps failing on recording an optimal obs."
				try:
					self.record_video(fig_path, desired)
					break
				except Exception as e:
					num_tries += 1
			#print(f"sequence to {self.problem_name} is:\n\t{steps}\ngenerating image at {img_path}.")
			print(f"generated sequence video at {fig_path}.")
		self.env.close()
		return observations

	# def reset_with_goal_idx(self, goal_idx):
	# 	self.env.set_options({"goal_idx": goal_idx})
	# 	return self.env.reset()
	
class GCDeepRLAgent(DeepRLAgent):
	def generate_partial_observation(self, action_selection_method, percentage, is_consecutive, goal_directed_problem=None, goal_directed_goal=None, save_fig=False, fig_path=None, random_optimalism=True):
		steps = self.generate_observation(action_selection_method, save_fig=save_fig, fig_path=fig_path, random_optimalism=random_optimalism, goal_directed_problem=goal_directed_problem, goal_directed_goal=goal_directed_goal) # steps are a full observation
		return random_subset_with_order(steps, (int)(percentage * len(steps)), is_consecutive)

	def generate_observation(self, action_selection_method: MethodType, random_optimalism, env_prop=None, goal_directed_problem=None, goal_directed_goal=None,
								save_fig = False, fig_path=None, with_dict=False):
		# print(f"hyperparams:{hyperparams}")
		if goal_directed_problem:
			if save_fig:
				assert fig_path != None, "You need to specify a vid path when you save the figure."
			else:
				assert fig_path == None
			assert goal_directed_goal == None, "can't give goal directed goal and also goal directed problem for the sake of sequence generation by a general agent"
			kwargs = {"id": goal_directed_problem, "render_mode": "rgb_array"}
			self.env = create_vec_env(kwargs)
			orig_env = self.env
			observations = super().generate_observation(action_selection_method=action_selection_method, random_optimalism=random_optimalism,
													 save_fig=save_fig, fig_path=fig_path, with_dict=with_dict)
			self.env = orig_env
		else: #goal_directed_goal!=None
			if save_fig:
				assert fig_path != None, "You need to specify a vid path when you save the figure."
			else:
				assert fig_path == None
			assert goal_directed_problem == None, "can't give goal directed goal and also goal directed problem for the sake of sequence generation by a general agent"
			observations = super().generate_observation(action_selection_method=action_selection_method, random_optimalism=random_optimalism,
											save_fig=save_fig, fig_path=fig_path, with_dict=with_dict, desired=goal_directed_goal) # TODO tutorial on how to use the deepRLAgent for sequence generation and examination and plotting of the sequence
		return observations
		

if __name__ == "__main__":
	package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	print("this is package root:" + package_root)
	if package_root not in sys.path:
		sys.path.insert(0, package_root)

	from gr_libs.ml.utils.storage import get_agent_model_dir, set_global_storage_configs

	set_global_storage_configs("graml", "fragmented_partial_obs", "inference_same_length", "learn_diff_length")
	agent = DeepRLAgent(domain_name="point_maze", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-9x1", algorithm=SAC, num_timesteps=200000)
	agent.learn()
	agent.record_video("")
