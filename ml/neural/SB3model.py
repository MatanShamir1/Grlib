from types import MethodType
import gymnasium as gym
import numpy as np
import cv2

if __name__ != "__main__":
    from ml.utils.storage import get_agent_model_dir, get_policy_sequences_result_path
    from ml.utils.format import random_subset_with_order

from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gr_libs.custom_env_wrappers.flat_obs_wrapper import CombineAchievedGoalAndObservationWrapper

# important for registration of envs! do not remove lad
import gr_libs.maze_scripts.envs.maze
import gr_libs.highway_env_scripts.envs.parking_env

# built-in python modules
import random
import os
import sys
import traceback
import inspect

def amplify(values, alpha=1.0):
    """Computes amplified softmax probabilities for an array of values
    Args:
        values (list): Input values for which to compute softmax
        alpha (float): Amplification factor, where alpha > 1 increases differences between probabilities
    Returns:
        np.array: amplified softmax probabilities
    """
    values = values[:3]**alpha # currently only choose to turn or move forward
    return values / np.sum(values)

def stochastic_amplified_selection(actions_probs, alpha=15.0):
    action_probs_amplified = amplify(actions_probs, alpha)
    choice = np.random.choice(len(action_probs_amplified), p=action_probs_amplified)
    if choice == 3:
        choice = 6
    return choice

class NeuralAgent():
    def __init__(self, env_name: str, problem_name: str, algorithm, num_timesteps:float, reward_threshold: float=450, exploration_rate=None, tasks_to_complete=None, complex_obs_space=False):
        # Need to change reward threshold to change according to which task the agent is training on, becuase it changes from task to task.
        kwargs = {"id":problem_name, "render_mode":"rgb_array"}
        if tasks_to_complete: kwargs["tasks_to_complete"] = tasks_to_complete
        env = gym.make(**kwargs)
        # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        # env = Monitor(env, "logs/", allow_early_resets=True)
        self.env_name = env_name
        self.problem_name = problem_name
        if tasks_to_complete and len(tasks_to_complete): problem_name += "".join(task+"_" for task in tasks_to_complete)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        if complex_obs_space: env = CombineAchievedGoalAndObservationWrapper(env)
        self.env = DummyVecEnv([lambda: env])
        self._actions_space = self.env.action_space
        if exploration_rate != None: self._model = algorithm("MultiInputPolicy", self.env, ent_coef=exploration_rate, verbose=1)
        else: self._model = algorithm("MultiInputPolicy", self.env, verbose=1)
        
        self._model_directory = get_agent_model_dir(model_name=problem_name, class_name=algorithm.__name__)
        self._model_file_path = os.path.join(self._model_directory, "saved_model.pth")
        self.algorithm = algorithm
        self.reward_threshold = reward_threshold
        self.num_timesteps = num_timesteps
        self.is_gc = problem_name == "parking-v0"
        
    def save_model(self):
        self._model.save(self._model_file_path)
        
    def record_video(self, video_path):
        """Record a video of the agent's performance."""
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 30.0
        self.env.reset()
        frame_size = (self.env.render(mode='rgb_array').shape[1], self.env.render(mode='rgb_array').shape[0])
        video_path = os.path.join(video_path, "plan_video.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        done = False
        obs = self.env.reset()
        counter = 0
        while not done:
            counter += 1
            action, _states = self._model.predict(obs, deterministic=True)
            obs, rewards, done, info = self.env.step(action)
            if done[0] == True:
                pass
            if "success" in info[0].keys(): assert done == info[0]["success"] # make sure the agent actually reached the goal within the max time
            elif "is_success" in info[0].keys(): assert done == info[0]["is_success"] # make sure the agent actually reached the goal within the max time
            elif "step_task_completions" in info[0].keys(): assert done[0] == (len(info[0]["step_task_completions"]) == 1) # bug of dummyVecEnv, it removes the episode_task_completions from the info dict.
            else: raise NotImplementedError("no other option for any of the environments.")
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
        elif self.algorithm == TD3:
            assert False
        else:
            assert False
        return actor_means, log_std_dev

    # fits agents that generated observations in the form of: list of tuples, each tuple a single step\frame with size 2, comprised of obs and action.
    # the function squashes the 2d array of obs and action in a 1d array, concatenating their values together for training.
    def simplify_observation(self, observation):
        return [np.concatenate((np.array(obs).reshape(obs.shape[-1]),np.array(action).reshape(action.shape[-1]))) for (obs,action) in observation]

    def generate_partial_observation(self, action_selection_method, percentage, is_fragmented, save_fig=False, random_optimalism=True):
        steps = self.generate_observation(action_selection_method, save_fig=save_fig, random_optimalism=random_optimalism) # steps are a full observation
        return random_subset_with_order(steps, (int)(percentage * len(steps)), is_fragmented)

    def generate_observation(self, action_selection_method: MethodType, random_optimalism, save_fig = False, specific_vid_name=None, with_dict=False):
        if save_fig == False:
            assert specific_vid_name == None, "You can't specify a vid path when you don't even save the figure."
        obs = self.env.reset()
        observations = []
        is_successful_observation_made = False
        num_of_insuccessful_attempts = 0
        while not is_successful_observation_made:
            is_successful_observation_made = True # start as true, if this isn't the case (crash/death/truncation instead of success)
            if random_optimalism:
                constant_initial_action = self.env.action_space.sample()
            while True:
                deterministic = action_selection_method != stochastic_amplified_selection
                action, _states = self._model.predict(obs, deterministic=deterministic)
                if random_optimalism : # get the right direction and then start inserting noise to still get a relatively optimal plan
                    if len(observations) > 5:
                        for i in range(0, len(action[0])):
                            action[0][i] += random.uniform(-0.1 * action[0][i], 0.1 * action[0][i])
                    else: # just walk in a specific random direction to enable diverse plans
                        action = np.array(np.array([constant_initial_action]), None)
                if with_dict: observations.append((obs, action))
                else: observations.append((obs['observation'], action))
                obs, reward, done, info = self.env.step(action)
                general_done = done[0]
                if "success" in info[0].keys(): success_done = info[0]["success"]
                elif "is_success" in info[0].keys(): success_done = info[0]["is_success"]
                elif "step_task_completions" in info[0].keys(): success_done = info[0]["step_task_completions"]
                else: raise NotImplementedError("no other option for any of the environments.")
                if general_done == True and success_done == False:
                    # it could be that the stochasticity inserted into the actions made the agent die/crash. we don't want this observation.
                    num_of_insuccessful_attempts += 1
                    # print(f"for agent for problem {self.problem_name}, its done {len(observations)} steps, and got to a situation where general_done != success_done, for the {num_of_insuccessful_attempts} time.")
                    if num_of_insuccessful_attempts > 50:
                        # print(f"got more then 10 insuccessful attempts. fuak!")
                        assert general_done == success_done # we want to make sure the episode is done only when the agent has actually succeeded with the task.
                    else:
                        # try again by breaking inner loop. everything is set up to be like the beginning of the function.
                        is_successful_observation_made = False
                        obs = self.env.reset()
                        observations = [] # we want to re-accumulate the observations from scratch, have another try
                        break
                elif general_done == False and success_done == False:
                    continue
                elif general_done == True and success_done == True:
                    if num_of_insuccessful_attempts > 0:
                        pass # print(f"after {num_of_insuccessful_attempts}, finally I succeeded!")
                    break
                elif general_done == False and success_done == True:
                    assert False # shouldn't happen
        if save_fig:
            vid_path = os.path.abspath(os.path.join(get_policy_sequences_result_path(self.env_name), self.problem_name))
            if specific_vid_name: vid_path += f"_{specific_vid_name}"
            if not os.path.exists(vid_path):
                os.makedirs(vid_path)
            num_tries = 0
            while True:
                if num_tries >= 10:
                    assert False, "agent keeps failing on recording an optimal obs."
                try:
                    self.record_video(vid_path)
                    break
                except Exception as e:
                    num_tries += 1
            #print(f"sequence to {self.problem_name} is:\n\t{steps}\ngenerating image at {img_path}.")
            print(f"generated sequence video at {vid_path}.")
        return observations
    
    def reset_with_goal_idx(self, goal_idx):
        self.env.set_options({"goal_idx": goal_idx})
        return self.env.reset()

    def generate_partial_observation_gc(self, action_selection_method, percentage, is_fragmented, goal_idx, save_fig=False, random_optimalism=True):
        assert self.is_gc
        steps = self.generate_observation_gc(action_selection_method, save_fig=save_fig, random_optimalism=random_optimalism, goal_idx=goal_idx) # steps are a full observation
        return random_subset_with_order(steps, (int)(percentage * len(steps)), is_fragmented)

    def generate_observation_gc(self, action_selection_method: MethodType, random_optimalism, goal_idx, save_fig = False):
        obs = self.reset_with_goal_idx(goal_idx)
        observations = []
        is_successful_observation_made = False
        num_of_insuccessful_attempts = 0
        while not is_successful_observation_made:
            is_successful_observation_made = True # start as true, if this isn't the case (crash/death/truncation instead of success)
            if random_optimalism:
                constant_initial_action = self.env.action_space.sample()
            while True:
                deterministic = action_selection_method != stochastic_amplified_selection
                action, _states = self._model.predict(obs, deterministic=deterministic)
                if random_optimalism : # get the right direction and then start inserting noise to still get a relatively optimal plan
                    if len(observations) > 5:
                        for i in range(0, len(action[0])):
                            action[0][i] += random.uniform(-0.1 * action[0][i], 0.1 * action[0][i])
                    else: # just walk in a specific random direction to enable diverse plans
                        action = constant_initial_action
                # obs, reward, done, info = self.env.step(action)
                observations.append((obs['observation'], action))
                obs, reward, done, info = self.env.step(action)
                general_done = done[0]
                if "success" in info[0].keys(): success_done = info[0]["success"]
                elif "is_success" in info[0].keys(): success_done = info[0]["is_success"]
                elif "step_task_completions" in info[0].keys(): success_done = info[0]["step_task_completions"]
                else: raise NotImplementedError("no other option for any of the environments.")
                if general_done == True and success_done == False:
                    # it could be that the stochasticity inserted into the actions made the agent die/crash. we don't want this observation.
                    num_of_insuccessful_attempts += 1
                    # print(f"for agent for problem {self.problem_name}, its done {len(observations)} steps, and got to a situation where general_done != success_done, for the {num_of_insuccessful_attempts} time.")
                    if num_of_insuccessful_attempts > 10:
                        # print(f"got more then 10 insuccessful attempts. fuak!")
                        assert general_done == success_done # we want to make sure the episode is done only when the agent has actually succeeded with the task.
                    else:
                        # try again by breaking inner loop. everything is set up to be like the beginning of the function.
                        is_successful_observation_made = False
                        obs = self.reset_with_goal_idx(goal_idx)
                        observations = [] # we want to re-accumulate the observations from scratch, have another try
                        break
                elif general_done == False and success_done == False:
                    continue
                elif general_done == True and success_done == True:
                    if num_of_insuccessful_attempts > 0:
                        pass# print(f"after {num_of_insuccessful_attempts}, finally I succeeded!")
                    break
                elif general_done == False and success_done == True:
                    assert False # shouldn't happen
        if save_fig:
            vid_path = os.path.abspath(os.path.join(get_policy_sequences_result_path(self.env_name), self.problem_name))
            if not os.path.exists(vid_path):
                os.makedirs(vid_path)
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

    from ml.utils.storage import get_agent_model_dir, set_global_storage_configs

    set_global_storage_configs("graml", "fragmented_partial_obs", "inference_same_length", "learn_diff_length")
    dynamic_goals = ['(7,3)', '(3,7)', '(6,4)', '(4,6)', '(4,4)', '(3,4)', '(7,7)', '(6,7)']
    # agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-7x3", algorithm=TD3, num_timesteps=200000)
    # agent.learn() # yes
    # agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-3x7", algorithm=TD3, num_timesteps=400000)
    # agent.learn() # yes
    # agent = NeuralAgent(env_name="PointMaze-FourRoomsEnv-11x11", problem_name="PointMaze-FourRoomsEnv-11x11-Goal-6x4", algorithm=TD3, num_timesteps=500000)
    # agent.learn() # no
    # agent = NeuralAgent(env_name="PointMaze-FourRoomsEnv-11x11", problem_name="PointMaze-FourRoomsEnv-11x11-Goal-4x6", algorithm=TD3, num_timesteps=500000)
    # agent.learn() # no
    # agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-4x4", algorithm=TD3, num_timesteps=200000)
    # agent.learn() # yes
    # agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-3x4", algorithm=TD3, num_timesteps=200000)
    # agent.learn() # yes

    # agent = NeuralAgent(env_name="FrankaKitchenEnv", problem_name="FrankaKitchen-v1", tasks_to_complete = ["kettle"], algorithm=SAC, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # num_timesteps = 400000
    # while True:
    #     agent = NeuralAgent(env_name="FrankaKitchenEnv", problem_name="FrankaKitchen-v1", tasks_to_complete = ["slide cabinet"], algorithm=SAC, num_timesteps=400000, complex_obs_space=True)
    #     agent.learn()
    #     if agent.env.is_success_once:
    #         break
    #     else:
    #         num_timesteps += 100000
    
    # agent = NeuralAgent(env_name="FrankaKitchenEnv", problem_name="FrankaKitchen-v1", tasks_to_complete = ["slide cabinet"], algorithm=SAC, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchenEnv", problem_name="FrankaKitchen-v1", tasks_to_complete = ["kettle"], algorithm=SAC, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchenEnv", problem_name="FrankaKitchen-v1", tasks_to_complete = ["bottom burner"], algorithm=SAC, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    agent = NeuralAgent(env_name="FrankaKitchenEnv", problem_name="FrankaKitchen-v1", tasks_to_complete = ["light switch"], algorithm=SAC, num_timesteps=400000, complex_obs_space=True)
    agent.learn()
    # num_timesteps = 400000
    # while True:
    #     agent = NeuralAgent(env_name="FrankaKitchenEnv", problem_name="FrankaKitchen-v1", tasks_to_complete = ["bottom burner"], algorithm=SAC, num_timesteps=400000, complex_obs_space=True)
    #     agent.learn()
    #     if agent.env.is_success_once:
    #         break
    #     else:
    #         num_timesteps += 100000
    # agent = NeuralAgent(env_name="FrankaKitchenEnv", problem_name="FrankaKitchen-v1", tasks_to_complete = ["bottom burner"], algorithm=SAC, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchenEnv", problem_name="FrankaKitchen-v1", tasks_to_complete = ["light switch"], algorithm=SAC, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["bottom burner", "top burner"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["light switch", "microwave"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["slide cabinet", "hinge cabinet"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["bottom burner", "microwave"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["kettle", "light switch"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["bottom burner", "hinge cabinet"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    
    # agent = NeuralAgent(env_name="ParkingEnv", problem_name="Parking-S-14-PC--GI-7-v0", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="ParkingEnv", problem_name="Parking-S-10-PC--GI-3-v0", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="ParkingEnv", problem_name="Parking-S-10-PC--GI-6-v0", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="ParkingEnv", problem_name="Parking-S-10-PC--GI-8-v0", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="ParkingEnv", problem_name="Parking-S-10-PC--GI-5-v0", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="ParkingEnv", problem_name="Parking-S-10-PC--GI-7-v0", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="ParkingEnv", problem_name="Parking-S-10-PC--GI-7-v0", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="parking-v0", problem_name="parking-v0", algorithm=TD3, num_timesteps=400000)
    # agent.learn()
    
    # agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-7x7", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="PointMaze-FourRoomsEnvDense-11x11", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-6x7", algorithm=SAC, num_timesteps=100000)
    # agent.learn()
    #print(os.path.join(GRAML_itself, "dataset/Videos/maze_video.mp4"))
    # agent.generate_full_observation()

    # [(1, 5), (5, 1), (3, 6), (6, 3), (5, 5), (9, 9)]
    # agent = NeuralAgent(env_name="PointMaze-ObstaclesEnv-11x11", problem_name="PointMaze-ObstaclesEnvDense-11x11-Goal-9x9", algorithm=SAC, num_timesteps=250000)
    # agent.learn()
    # agent = NeuralAgent(env_name="PointMaze-ObstaclesEnv-11x11", problem_name="PointMaze-ObstaclesEnvDense-11x11-Goal-5x5", algorithm=SAC, num_timesteps=250000)
    # agent.learn()
    # agent = NeuralAgent(env_name="PointMaze-ObstaclesEnv-11x11", problem_name="PointMaze-ObstaclesEnvDense-11x11-Goal-5x5", algorithm=TD3, num_timesteps=250000)
    # agent.learn()
    # agent = NeuralAgent(env_name="PointMaze-ObstaclesEnv-11x11", problem_name="PointMaze-ObstaclesEnvDense-11x11-Goal-3x6", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="PointMaze-ObstaclesEnv-11x11", problem_name="PointMaze-ObstaclesEnvDense-11x11-Goal-6x3", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="PointMaze-ObstaclesEnv-11x11", problem_name="PointMaze-ObstaclesEnvDense-11x11-Goal-1x5", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    # agent = NeuralAgent(env_name="PointMaze-ObstaclesEnv-11x11", problem_name="PointMaze-ObstaclesEnvDense-11x11-Goal-5x1", algorithm=SAC, num_timesteps=200000)
    # agent.learn()
    
    agent.record_video("")
    
#     OBS_ELEMENT_GOALS = {
#     "bottom burner": np.array([-0.88, -0.01]),
#     "top burner": np.array([-0.92, -0.01]),
#     "light switch": np.array([-0.69, -0.05]),
#     "slide cabinet": np.array([0.37]),
#     "hinge cabinet": np.array([0.0, 1.45]),
#     "microwave": np.array([-0.75]),
#     "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
# }
    