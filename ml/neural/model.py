from types import MethodType
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gr_libs.maze_scripts.envs.maze
import os
import sys
import traceback
import inspect
import cv2
if __name__ != "__main__":
    from ml.utils.format import random_subset_with_order

class NeuralAgent():
    def __init__(self, problem_name: str, algorithm, reward_threshold: float=450):
        env = gym.make(problem_name, render_mode="rgb_array")
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env, "logs/", allow_early_resets=True)
        self.env = DummyVecEnv([lambda: env])
        self._actions_space = self.env.action_space
        self._model = algorithm("MultiInputPolicy", self.env, verbose=1)
        self._model_directory = "dataset/sac_maze_agent"
        self._model_file_path = os.path.join(self._model_directory, f"{problem_name}.pth")
        self.algorithm = algorithm
        self.reward_threshold = reward_threshold
        
    def save_model(self):
        self._model.save(self._model_file_path)
        
    def record_video(self, video_path, video_length=500):
        """Record a video of the agent's performance."""
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 30.0
        self.env.reset()
        frame_size = (self.env.render().shape[1], self.env.render().shape[0])
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        
        obs = self.env.reset()
        for _ in range(video_length):
            action, _states = self._model.predict(obs, deterministic=True)
            obs, rewards, dones, info = self.env.step(action)
            frame = self.env.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if dones:
                obs = self.env.reset()
        
        video_writer.release()
    
    def load_model(self):
        self._model = self.algorithm.load(self._model_file_path)
    
    def learn(self):
        if os.path.exists(self._model_file_path):
            print(f"Loading pre-existing model in {self._model_file_path}")
            self.load_model()
        else:
            # Stop training when the model reaches the reward threshold
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=self.reward_threshold, verbose=1)
            eval_callback = EvalCallback(self.env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500, callback_on_new_best=callback_on_best, verbose=1, render=True)
            self._model.learn(total_timesteps=100000, progress_bar=True, callback=eval_callback)
            self.save_model()

    def generate_partial_observation(self, action_selection_method, percentage, is_fragmented, save_fig=True, random_optimalism=True):
        steps = self.generate_observation(action_selection_method, save_fig, random_optimalism) # steps are a full observation
        return random_subset_with_order(steps, (int)(percentage * len(steps)), is_fragmented)

    def generate_observation(self, action_selection_method: MethodType, random_optimalism, save_fig = False):
        obs = self.env.reset()
        observations = []
        is_done = False
        try:
            while not is_done:
                action, _ = self._model.predict(obs, deterministic=not random_optimalism)
                # obs, reward, done, info = self.env.step(action)
                observations.append(obs['observation'])
                obs, reward, done, info = self.env.step(action)
                is_done = info[0]["is_success"]
                assert done[0] == is_done
                self.env.render()
        except BaseException as e:
            print("An exception occurred: fucking shait")
            traceback.print_exc()
        print(observations)
        print(f'len of observations: {len(observations)}')
        

if __name__ == "__main__":
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("this is package root:" + package_root)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from ml.utils.storage import get_model_dir, problem_list_to_str_tuple
    # agent = NeuralAgent("PandaReachSimple-g-m01xm01-v3", "PandaReachSimple-g-m01xm01-v3")
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    GRAML_itself = os.path.dirname(currentdir)
    GRAML_includer = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0, GRAML_includer)
    sys.path.insert(0, GRAML_itself)

    from GRAML.ml.utils.storage import set_global_storage_configs

    set_global_storage_configs("graql", "continuing_partial_obs")
    agent = NeuralAgent(problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-9x9", algorithm=SAC, reward_threshold=450)
    agent.learn()
    print(os.path.join(GRAML_itself, "dataset/Videos/maze_video.mp4"))
    # agent.generate_full_observation()
    agent.record_video("maze_video.mp4")
    