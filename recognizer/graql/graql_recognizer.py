import os
import dill
from abc import ABC
from types import MethodType
from typing import List, Tuple, Type
from gymnasium.spaces.discrete import Discrete
import ml
from ml import utils
from ml.base import RLAgent
from environment.utils import goal_str_to_tuple
from ml.utils.storage import get_graql_experiment_confidence_path


class GraqlRecognizer(ABC):
    def __init__(self, method: Type[RLAgent], env_name: str, problems: List[str], evaluation_function,
                 specified_rl_algorithm, collect_statistics=True, train_configs=None, task_str_to_goal=None):
        self.problems = problems
        self.env_name = env_name
        self.rl_agents_method = method
        self.task_str_to_goal = task_str_to_goal
        self.active_goals = [self.task_str_to_goal(problem_name) for problem_name in problems]
        self.agents = {}
        self.collect_statistics = collect_statistics
        self.specified_rl_algorithm = specified_rl_algorithm
        assert train_configs != None
        self.train_configs = train_configs
        self.evaluation_function = evaluation_function

    def domain_learning_phase(self):
        for i, problem_name in enumerate(self.problems):
            agent = self.rl_agents_method(env_name=self.env_name, problem_name=problem_name,
                                          num_timesteps=self.train_configs[i][1], algorithm=self.specified_rl_algorithm)
            agent.learn()
            self.agents[self.task_str_to_goal(problem_name)] = agent
            self.action_space = agent.env.action_space

    def goals_adaptation_phase(self, dynamic_goals_problems, dynamic_train_configs):
        self.active_goals =  [self.task_str_to_goal(problem) for problem in dynamic_goals_problems]
        for i, (problem_name, goal) in enumerate(zip(dynamic_goals_problems, self.active_goals)):
            agent = self.rl_agents_method(env_name=self.env_name, problem_name=problem_name,
                                          num_timesteps=dynamic_train_configs[i][1], algorithm=self.specified_rl_algorithm)
            agent.learn()
            self.agents[goal] = agent

    def inference_phase(self, inf_sequence, true_goal, percentage) -> str:
        scores = []
        for goal in self.active_goals:
            score = self.evaluation_function(inf_sequence, self.agents[goal], self.action_space)
            scores.append(score)
        #scores = metrics.softmin(np.array(scores))
        if self.collect_statistics:
            results_path = get_graql_experiment_confidence_path(self.env_name)
            if not os.path.exists(results_path): os.makedirs(results_path)
            with open(results_path + f'/true_{true_goal}_{percentage}_scores.pkl', 'wb') as scores_file:
                dill.dump([(str(goal), score) for (goal, score) in zip(self.active_goals, scores)], scores_file)
        div, true_goal_index = min((div, goal) for (goal, div) in enumerate(scores))
        return str(self.active_goals[true_goal_index])
