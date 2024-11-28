import os
import dill
from abc import ABC
from types import MethodType
from typing import List, Tuple, Type
from gymnasium.spaces.discrete import Discrete
import numpy as np
from stable_baselines3 import SAC, TD3
from grlib.ml import utils
from grlib.ml.base import RLAgent
from grlib.environment.utils import goal_str_to_tuple
from grlib.ml.utils.storage import get_graql_experiment_confidence_path


class GraqlRecognizer(ABC):
	def __init__(self, method: Type[RLAgent], env_name: str, problems: List[str], evaluation_function,
				 collect_statistics=True, train_configs=None, task_str_to_goal=None, is_universal=False, use_goal_directed_problem=None):
		self.is_universal = is_universal
		if is_universal:
			assert len(problems) == 1, "you should only give 1 problem to the universal goal recognizer, like parking-v0, etc."
		self.problems = problems
		self.env_name = env_name
		self.rl_agents_method = method
		self.task_str_to_goal = task_str_to_goal
		if is_universal: self.active_goals = None
		else: self.active_goals = [self.task_str_to_goal(problem_name) for problem_name in problems]
		self.agents = {} # consider changing to ContextualAgent
		self.collect_statistics = collect_statistics
		assert train_configs != None
		self.train_configs = train_configs
		self.evaluation_function = evaluation_function
		self.use_goal_directed_problem = use_goal_directed_problem

	def domain_learning_phase(self):
		for i, problem_name in enumerate(self.problems):
			agent_kwargs = {"env_name": self.env_name,
							"problem_name": problem_name}
			if self.train_configs[i][0] != None: agent_kwargs["algorithm"] = self.train_configs[i][0]
			if self.train_configs[i][1] != None: agent_kwargs["num_timesteps"] = self.train_configs[i][1]
			agent = self.rl_agents_method(**agent_kwargs)
			agent.learn()
			self.agents[self.task_str_to_goal(problem_name)] = agent
			self.action_space = agent.env.action_space

	def goals_adaptation_phase(self, dynamic_goals_problems, dynamic_train_configs):
		self.active_goals =  [self.task_str_to_goal(problem) for problem in dynamic_goals_problems]
		if self.is_universal: return
		for i, (problem_name, goal) in enumerate(zip(dynamic_goals_problems, self.active_goals)):
			agent_kwargs = {"env_name": self.env_name,
							"problem_name": problem_name}
			if dynamic_train_configs[i][0]: agent_kwargs["algorithm"] = dynamic_train_configs[i][0]
			if dynamic_train_configs[i][1]: agent_kwargs["num_timesteps"] = dynamic_train_configs[i][1]
			agent = self.rl_agents_method(**agent_kwargs)
			agent.learn()
			self.agents[goal] = agent

	def inference_phase(self, inf_sequence, true_goal, percentage) -> str:
		scores = []
		for goal in self.active_goals:
			if self.is_universal:
				agent = next(iter(self.agents.values()))
				if not self.use_goal_directed_problem:
					for obs in inf_sequence:
						obs[0]['desired_goal'] = np.array([goal], dtype=obs[0]['desired_goal'].dtype)
			else:
				agent = self.agents[goal]
			score = self.evaluation_function(inf_sequence, agent, self.action_space)
			scores.append(score)
		#scores = metrics.softmin(np.array(scores))
		if self.collect_statistics:
			results_path = get_graql_experiment_confidence_path(self.env_name)
			if not os.path.exists(results_path): os.makedirs(results_path)
			with open(results_path + f'/true_{true_goal}_{percentage}_scores.pkl', 'wb') as scores_file:
				dill.dump([(str(goal), score) for (goal, score) in zip(self.active_goals, scores)], scores_file)
		div, true_goal_index = min((div, goal) for (goal, div) in enumerate(scores))
		return str(self.active_goals[true_goal_index])
