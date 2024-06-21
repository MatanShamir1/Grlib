import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import os
import panda_gym
import sys


class PPOAgent():
    def __init__(self, env_name: str, problem_name: str):
        self._env = make_vec_env(problem_name)
        # self._env = make_vec_env(problem_name, n_envs=1)
        self._actions_space = self._env.action_space
        self._model = PPO("MultiInputPolicy", self._env, verbose=1)
        self._model_directory = get_model_dir(env_name=problem_name, model_name=problem_name, class_name=self.__class__.__name__)
        self._model_file_path = os.path.join(self._model_directory, "ppo_agent.pth")
        
    def save_model(self):
        self._model.save(self._model_file_path)
    
    def load_model(self):
        self._model = PPO.load(self._model_file_path)
    
    def learn(self):
        if os.path.exists(self._model_file_path):
            print(f"Loading pre-existing ppo model in {self._model_file_path}")
            self.load_model()
        else:
            self._model.learn(total_timesteps=2000000)
            self.save_model()

    def generate_full_observation(self):
        obs = self._env.reset()
        observations = [obs]
        done = False
        while not done:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, reward, done, info = self._env.step(action)
            observations.append(obs)
            self._env.render()
    
    def generate_partial_observation(self):
        pass

if __name__ == "__main__":
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from ml.utils.storage import get_model_dir, problem_list_to_str_tuple
    agent = PPOAgent("PandaPushSimple-g-m01xm01-o-01x01-v3", "PandaPushSimple-g-m01xm01-o-01x01-v3")
    agent.learn()
    agent.generate_full_observation()
    