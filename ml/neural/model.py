from types import MethodType
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import gr_libs.maze_scripts.envs.maze
import os
import sys
import traceback
import inspect
import cv2

if __name__ != "__main__":
    from ml.utils.storage import get_model_dir, get_policy_sequences_result_path
    from metrics.metrics import stochastic_amplified_selection
    from ml.utils.format import random_subset_with_order

class NeuralAgent():
    def __init__(self, env_name: str, problem_name: str, algorithm, num_timesteps:float, reward_threshold: float=450, exploration_rate=None):
        # Need to change reward threshold to change according to which task the agent is training on, becuase it changes from task to task.
        env = gym.make(problem_name, render_mode="rgb_array")
        # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        # env = Monitor(env, "logs/", allow_early_resets=True)
        self.env_name = env_name
        self.problem_name = problem_name
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        self.env = DummyVecEnv([lambda: env])
        self._actions_space = self.env.action_space
        if exploration_rate != None: self._model = algorithm("MultiInputPolicy", self.env, ent_coef=exploration_rate, verbose=1)
        else: self._model = algorithm("MultiInputPolicy", self.env, verbose=1)
        self._model_directory = get_model_dir(env_name=env_name, model_name=problem_name, class_name=algorithm.__name__)
        self._model_file_path = os.path.join(self._model_directory, "saved_model.pth")
        self.algorithm = algorithm
        self.reward_threshold = reward_threshold
        self.num_timesteps = num_timesteps
        
    def save_model(self):
        self._model.save(self._model_file_path)
        
    def record_video(self, video_path):
        """Record a video of the agent's performance."""
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 30.0
        self.env.reset()
        frame_size = (self.env.render(mode='rgb_array').shape[1], self.env.render(mode='rgb_array').shape[0])
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        done = False
        obs = self.env.reset()
        while not done:
            action, _states = self._model.predict(obs, deterministic=True)
            obs, rewards, done, info = self.env.step(action)
            assert done == info[0]["success"] # make sure the agent actually reached the goal within the max time
            frame = self.env.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
    
    def load_model(self):
        self._model = self.algorithm.load(self._model_file_path)
    
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
            self._model.learn(total_timesteps=self.num_timesteps, progress_bar=True) # comment this in a normal env
            self.save_model()

    def simplify_observation(self, observation):
        return [np.concatenate((np.array(obs).reshape(obs.shape[-1]),np.array(action).reshape(action.shape[-1]))) for (obs,action) in observation]

    def generate_partial_observation(self, action_selection_method, percentage, is_fragmented, save_fig=False, random_optimalism=True):
        steps = self.generate_observation(action_selection_method, save_fig=save_fig, random_optimalism=random_optimalism) # steps are a full observation
        return random_subset_with_order(steps, (int)(percentage * len(steps)), is_fragmented)

    def generate_observation(self, action_selection_method: MethodType, random_optimalism, save_fig = False):
        obs = self.env.reset()
        observations = []
        is_done = False
        while not is_done:
            deterministic = action_selection_method != stochastic_amplified_selection and not random_optimalism
            action, _states = self._model.predict(obs, deterministic=deterministic)
            # obs, reward, done, info = self.env.step(action)
            observations.append((obs['observation'], action))
            obs, reward, done, info = self.env.step(action)
            is_done = info[0]["success"]
            assert done[0] == is_done # we want to make sure the episode is done only when the agent has actually succeeded with the task.
        #print(f'len of observations: {len(observations)}')
        if save_fig:
            vid_path = os.path.join(get_policy_sequences_result_path(self.env_name), self.problem_name)
            self.record_video(vid_path)
            #print(f"sequence to {self.problem_name} is:\n\t{steps}\ngenerating image at {img_path}.")
            print(f"generated sequence video at {vid_path}.")
        return observations
        

if __name__ == "__main__":
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("this is package root:" + package_root)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    # agent = NeuralAgent("PandaReachSimple-g-m01xm01-v3", "PandaReachSimple-g-m01xm01-v3")
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    GRAML_itself = os.path.dirname(currentdir)
    GRAML_includer = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0, GRAML_includer)
    sys.path.insert(0, GRAML_itself)
    
    from ml.utils.storage import get_model_dir, set_global_storage_configs

    set_global_storage_configs("graml", "fragmented_partial_obs", "inference_same_length", "learn_diff_length")
    dynamic_goals = ['(7,3)', '(3,7)', '(6,4)', '(4,6)', '(4,4)', '(3,4)', '(7,7)']
    agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-7x3", algorithm=SAC, num_timesteps=200000)
    agent.learn()
    agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-3x7", algorithm=SAC, num_timesteps=200000)
    agent.learn()
    agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-6x4", algorithm=SAC, num_timesteps=200000)
    agent.learn()
    agent = NeuralAgent(env_name="PointMaze-FourRoomsEnv-11x11", problem_name="PointMaze-FourRoomsEnv-11x11-Goal-4x6", algorithm=SAC, num_timesteps=500000)
    agent.learn()
    agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-4x4", algorithm=SAC, num_timesteps=200000)
    agent.learn()
    agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-3x4", algorithm=SAC, num_timesteps=200000)
    agent.learn()
    agent = NeuralAgent(env_name="PointMaze-FourRoomsEnv-11x11", problem_name="PointMaze-FourRoomsEnv-11x11-Goal-7x7", algorithm=SAC, num_timesteps=1000000)
    agent.learn()
    print(os.path.join(GRAML_itself, "dataset/Videos/maze_video.mp4"))
    # agent.generate_full_observation()
    agent.record_video("maze_video.mp4")
    