from abc import abstractmethod
import os
import dill
from typing import List, Type
import numpy as np
from gr_libs.environment.environment import EnvProperty, GCEnvProperty
from gr_libs.environment.utils.utils import domain_to_env_property
from gr_libs.metrics.metrics import kl_divergence_norm_softmax, mean_wasserstein_distance
from gr_libs.ml.base import RLAgent
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent, GCDeepRLAgent
from gr_libs.ml.tabular.tabular_q_learner import TabularQLearner
from gr_libs.ml.utils.storage import get_gr_as_rl_experiment_confidence_path
from gr_libs.recognizer.recognizer import GaAdaptingRecognizer, GaAgentTrainerRecognizer, LearningRecognizer, Recognizer

class GRAsRL(Recognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.agents = {} # consider changing to ContextualAgent

	def goals_adaptation_phase(self, dynamic_goals: List[str], dynamic_train_configs):
		super().goals_adaptation_phase(dynamic_goals, dynamic_train_configs)
		dynamic_goals_problems = [self.env_prop.goal_to_problem_str(goal) for goal in dynamic_goals]
		self.active_goals = dynamic_goals
		self.active_problems = dynamic_goals_problems
		for problem_name, config in zip(dynamic_goals_problems, dynamic_train_configs):
			agent_kwargs = {"domain_name": self.env_prop.domain_name,
							"problem_name": problem_name}
			if config[0]: agent_kwargs["algorithm"] = config[0]
			if config[1]: agent_kwargs["num_timesteps"] = config[1]
			agent = self.rl_agent_type(**agent_kwargs)
			agent.learn()
			self.agents[problem_name] = agent
		self.action_space = next(iter(self.agents.values())).env.action_space

	def inference_phase(self, inf_sequence, true_goal, percentage) -> str:
		scores = []
		for problem_name in self.active_problems:
			agent = self.choose_agent(problem_name)
			if self.env_prop.gc_adaptable():
				assert self.__class__.__name__ == "GCDraco", "This recognizer is not compatible with goal conditioned problems."
				inf_sequence = self.prepare_inf_sequence(problem_name, inf_sequence)
			score = self.evaluation_function(inf_sequence, agent, self.action_space)
			scores.append(score)
		#scores = metrics.softmin(np.array(scores))
		if self.collect_statistics:
			results_path = get_gr_as_rl_experiment_confidence_path(domain_name=self.env_prop.domain_name, env_name=self.env_prop.name, recognizer=self.__class__.__name__)
			if not os.path.exists(results_path): os.makedirs(results_path)
			with open(results_path + f'/true_{true_goal}_{percentage}_scores.pkl', 'wb') as scores_file:
				dill.dump([(str(goal), score) for (goal, score) in zip(self.active_goals, scores)], scores_file)
		div, true_goal_index = min((div, goal) for (goal, div) in enumerate(scores))
		return str(self.active_goals[true_goal_index])
	
	def choose_agent(self, problem_name:str) -> RLAgent:
		return self.agents[problem_name]


class Graql(GRAsRL, GaAgentTrainerRecognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert not self.env_prop.gc_adaptable() and self.env_prop.is_state_discrete() and self.env_prop.is_action_discrete()
		if self.rl_agent_type==None: self.rl_agent_type = TabularQLearner
		self.evaluation_function = kl_divergence_norm_softmax

class Draco(GRAsRL, GaAgentTrainerRecognizer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert not self.env_prop.is_state_discrete() and not self.env_prop.is_action_discrete()
		if self.rl_agent_type==None: self.rl_agent_type = DeepRLAgent
		self.evaluation_function = mean_wasserstein_distance

class GCDraco(GRAsRL, LearningRecognizer, GaAdaptingRecognizer): # TODO problem: it gets 2 goal_adaptation phase from parents, one with configs and one without.
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert self.env_prop.gc_adaptable() and not self.env_prop.is_state_discrete() and not self.env_prop.is_action_discrete()
		self.evaluation_function = mean_wasserstein_distance
		if self.rl_agent_type==None: self.rl_agent_type = GCDeepRLAgent

	def domain_learning_phase(self, base_goals: List[str], train_configs):
		super().domain_learning_phase(base_goals, train_configs)
		agent_kwargs = {"domain_name": self.env_prop.domain_name,
						"problem_name": self.env_prop.name,
						"algorithm": self.original_train_configs[0][0],
						"num_timesteps": self.original_train_configs[0][1]}
		agent = self.rl_agent_type(**agent_kwargs)
		agent.learn()
		self.agents[self.env_prop.name] = agent
		self.action_space = agent.env.action_space

	# this method currently does nothing but optimizations can be made here.
	def goals_adaptation_phase(self, dynamic_goals):
		self.active_goals = dynamic_goals
		self.active_problems = [self.env_prop.goal_to_problem_str(goal) for goal in dynamic_goals]
	
	def choose_agent(self, problem_name:str) -> RLAgent:
		return next(iter(self.agents.values()))
	
	def prepare_inf_sequence(self, problem_name: str, inf_sequence):
		if not self.env_prop.use_goal_directed_problem():
			for obs in inf_sequence:
				obs[0]['desired_goal'] = np.array([self.env_prop.str_to_goal(problem_name)], dtype=obs[0]['desired_goal'].dtype)
			return inf_sequence
		return inf_sequence
