import os
import dill
from abc import ABC
from types import MethodType
from typing import List, Tuple, Type
from gymnasium.spaces.discrete import Discrete

from metrics import metrics
import ml
from ml import utils
from ml.base import RLAgent
from ml.utils.format import goal_str_to_tuple, minigrid_str_to_goal
from ml.utils.storage import get_graql_experiment_confidence_path


class GraqlRecognizer(ABC):
    def __init__(self, method: Type[RLAgent], env_name: str, base_problems: List[str], goal_to_task_str:MethodType,
                 is_continuous = False, is_fragmented=True, is_inference_same_length_sequences=False, is_learn_same_length_sequences=False, collect_statistics=True):
        self.base_problems = base_problems
        self.env_name = env_name
        self.rl_agents_method = method
        self.active_goals = [minigrid_str_to_goal(problem_name) for problem_name in base_problems]
        self.agents = {}
        self.collect_statistics = collect_statistics
        self.goal_to_task_str = goal_to_task_str

    def domain_learning_phase(self):
        for problem_name in self.base_problems:
            agent = self.rl_agents_method(env_name=self.env_name, problem_name=problem_name)
            agent.learn()
            self.agents[minigrid_str_to_goal(problem_name)] = agent
            self.action_space = agent.env.action_space

    def goals_adaptation_phase(self, new_goals):
        self.active_goals = new_goals
        problems_goals = [(self.goal_to_task_str(tuply),tuply) for tuply in new_goals]
        for problem_name, goal_str in problems_goals:
            goal = goal_str_to_tuple(goal_str)
            agent = self.rl_agents_method(env_name=self.env_name, problem_name=problem_name)
            agent.learn()
            self.agents[goal] = agent

    def inference_phase(self, inf_sequence, true_goal, percentage, evaluation_function=metrics.kl_divergence_norm_softmax):
        scores = []
        for goal in self.active_goals:
            score = evaluation_function(inf_sequence, self.agents[goal_str_to_tuple(goal)], self.action_space)
            scores.append(score)
        #scores = metrics.softmin(np.array(scores))
        if self.collect_statistics:
            results_path = get_graql_experiment_confidence_path(self.env_name)
            if not os.path.exists(results_path): os.makedirs(results_path)
            with open(results_path + f'/true_{true_goal}_{percentage}_scores.pkl', 'wb') as scores_file:
                dill.dump([(goal, score) for (goal, score) in zip(self.active_goals, scores)], scores_file)
        div, true_goal_index = min((div, goal) for (goal, div) in enumerate(scores))
        return self.active_goals[true_goal_index]
