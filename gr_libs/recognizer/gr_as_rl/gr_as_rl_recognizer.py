import os

import dill
import numpy as np

from gr_libs.metrics.metrics import kl_divergence_norm_softmax
from gr_libs.ml.base import RLAgent
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent, GCDeepRLAgent
from gr_libs.ml.tabular.tabular_q_learner import TabularQLearner
from gr_libs.ml.utils.storage import get_gr_as_rl_experiment_confidence_path
from gymnasium.envs.registration import register, registry
from gr_libs.recognizer.recognizer import (
    GaAdaptingRecognizer,
    GaAgentTrainerRecognizer,
    LearningRecognizer,
    Recognizer,
)
from gr_libs.ml.consts import FINETUNE_TIMESTEPS


class GRAsRL(Recognizer):
    """
    GRAsRL class represents a goal recognition framework that using reinforcement learning.
    It inherits from the Recognizer class and implements the goal recognition process, including the
    Goal adaptation and the inference phase. It trains agents for each new goal, which makes it impractical
    for realtime environments where goals mmight change.

    Attributes:
        agents (dict): A dictionary that maps problem names to RLAgent instances.
        active_goals (List[str]): A list of active goals.
        active_problems (List[str]): A list of active problem names.
        action_space (gym.Space): The action space of the RLAgent.

    Methods:
        goals_adaptation_phase: Performs the goals adaptation phase.
        prepare_inf_sequence: Prepares the inference sequence for goal-directed problems.
        inference_phase: Performs the inference phase and returns the recognized goal.
        choose_agent: Returns the RLAgent for a given problem name.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agents = {}  # consider changing to ContextualAgent

    def goals_adaptation_phase(self, dynamic_goals: list[str], dynamic_train_configs):
        """
        Performs the goals adaptation phase.

        Args:
            dynamic_goals (List[str]): A list of dynamic goals.
            dynamic_train_configs: The dynamic training configurations.

        Returns:
            None
        """
        super().goals_adaptation_phase(dynamic_goals, dynamic_train_configs)
        dynamic_goals_problems = [
            self.env_prop.goal_to_problem_str(goal) for goal in dynamic_goals
        ]
        self.active_goals = dynamic_goals
        self.active_problems = dynamic_goals_problems
        for problem_name, config in zip(dynamic_goals_problems, dynamic_train_configs):
            agent_kwargs = {
                "domain_name": self.env_prop.domain_name,
                "problem_name": problem_name,
                "env_prop": self.env_prop,
            }
            if config[0]:
                agent_kwargs["algorithm"] = config[0]
            if config[1]:
                agent_kwargs["num_timesteps"] = config[1]
            agent = self.rl_agent_type(**agent_kwargs)
            agent.learn()
            self.agents[problem_name] = agent
        self.action_space = next(iter(self.agents.values())).env.action_space

    def prepare_inf_sequence(self, problem_name: str, inf_sequence):
        """
        Prepares the inference sequence for goal-directed problems.

        Args:
            problem_name (str): The name of the problem.
            inf_sequence: The inference sequence.

        Returns:
            The prepared inference sequence.
        """
        if not self.env_prop.use_goal_directed_problem():
            for obs in inf_sequence:
                obs[0]["desired_goal"] = np.array(
                    [self.env_prop.str_to_goal(problem_name)],
                    dtype=obs[0]["desired_goal"].dtype,
                )
            return inf_sequence
        return inf_sequence

    def inference_phase(self, inf_sequence, true_goal, percentage) -> str:
        """
        Performs the inference phase and returns the recognized goal.

        Args:
            inf_sequence: The inference sequence.
            true_goal: The true goal.
            percentage: The percentage.

        Returns:
            The recognized goal as a string.
        """
        scores = []
        for problem_name in self.active_problems:
            agent = self.choose_agent(problem_name)
            if self.env_prop.gc_adaptable():
                inf_sequence = self.prepare_inf_sequence(problem_name, inf_sequence)
            score = self.evaluation_function(inf_sequence, agent, self.action_space)
            scores.append(score)

        if self.collect_statistics:
            results_path = get_gr_as_rl_experiment_confidence_path(
                domain_name=self.env_prop.domain_name,
                env_name=self.env_prop.name,
                recognizer=self.__class__.__name__,
            )
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            with open(
                results_path + f"/true_{true_goal}_{percentage}_scores.pkl", "wb"
            ) as scores_file:
                dill.dump(
                    [
                        (str(goal), score)
                        for (goal, score) in zip(self.active_goals, scores)
                    ],
                    scores_file,
                )
        div, true_goal_index = min((div, goal) for (goal, div) in enumerate(scores))
        return str(self.active_goals[true_goal_index])

    def choose_agent(self, problem_name: str) -> RLAgent:
        """
        Returns the RLAgent for a given problem name.

        Args:
            problem_name (str): The name of the problem.

        Returns:
            The RLAgent instance.
        """
        return self.agents[problem_name]


class Graql(GRAsRL, GaAgentTrainerRecognizer):
    """
    Graql extends the GRAsRL framework and GaAgentTrainerRecognizer, since it trains new agents for every new goal and it adheres
    to the goal recognition as reinforcement learning framework. It uses a tabular Q-learning agent for discrete state and action spaces.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            not self.env_prop.gc_adaptable()
            and self.env_prop.is_state_discrete()
            and self.env_prop.is_action_discrete()
        )
        if self.rl_agent_type is None:
            self.rl_agent_type = TabularQLearner
        self.evaluation_function = kl_divergence_norm_softmax


class Draco(GRAsRL, GaAgentTrainerRecognizer):
    """
    Draco class represents a recognizer agent trained using the GRAsRL framework.
    Like Graql, it trains new agents for every new goal and adheres to the goal recognition as reinforcement learning framework.
    It uses a deep reinforcement learning agent for continuous state and action spaces.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        rl_agent_type (type): Type of the reinforcement learning agent.
        evaluation_function (callable): Function used for evaluation.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add any additional initialization code here

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            not self.env_prop.is_state_discrete()
            and not self.env_prop.is_action_discrete()
        )
        if self.rl_agent_type == None:
            self.rl_agent_type = DeepRLAgent
        self.evaluation_function = kwargs.get("evaluation_function")
        if self.evaluation_function is None:
            from gr_libs.metrics.metrics import mean_wasserstein_distance

            self.evaluation_function = mean_wasserstein_distance
        assert callable(
            self.evaluation_function
        ), "Evaluation function must be a callable function."


class GCDraco(GRAsRL, LearningRecognizer, GaAdaptingRecognizer):
    """
    GCDraco recognizer uses goal-conditioned reinforcement learning using the Draco algorithm.
    It inherits from GRAsRL, LearningRecognizer, and GaAdaptingRecognizer.
    It is designed for environments with continuous state and action spaces.
    It uses a goal-conditioned deep reinforcement learning agent for training and inference, which
    enables it to adapt to new goals during the goal adaptation phase without requiring retraining,
    making it suitable for dynamic environments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.env_prop.gc_adaptable()
            and not self.env_prop.is_state_discrete()
            and not self.env_prop.is_action_discrete()
        )
        if self.rl_agent_type == None:
            self.rl_agent_type = GCDeepRLAgent
        self.evaluation_function = kwargs.get("evaluation_function")
        if self.evaluation_function is None:
            from gr_libs.metrics.metrics import mean_wasserstein_distance

            self.evaluation_function = mean_wasserstein_distance
        assert callable(
            self.evaluation_function
        ), "Evaluation function must be a callable function."

    def domain_learning_phase(self, problems):
        base = problems["gc"]
        base_goals = base["goals"]
        train_configs = base["train_configs"]
        super().domain_learning_phase(train_configs, base_goals)
        agent_kwargs = {
            "domain_name": self.env_prop.domain_name,
            "problem_name": self.env_prop.name,
            "algorithm": self.original_train_configs[0][0],
            "num_timesteps": self.original_train_configs[0][1],
            "env_prop": self.env_prop,
        }
        agent = self.rl_agent_type(**agent_kwargs)
        agent.learn()
        self.agents[self.env_prop.name] = agent
        self.action_space = agent.env.action_space

    # this method currently does nothing but optimizations can be made here.
    def goals_adaptation_phase(self, dynamic_goals):
        self.active_goals = dynamic_goals
        self.active_problems = [
            self.env_prop.goal_to_problem_str(goal) for goal in dynamic_goals
        ]

    def choose_agent(self, problem_name: str) -> RLAgent:
        return next(iter(self.agents.values()))


class GCAura(GRAsRL, LearningRecognizer, GaAdaptingRecognizer):
    """
    GCAura uses goal-conditioned reinforcement learning with adaptive fine-tuning.

    It trains a base goal-conditioned policy over a goal subspace in the domain learning phase.
    During the goal adaptation phase, it checks if new goals are within the original goal subspace:
    - If a goal is within the subspace, it uses the original trained model
    - If a goal is outside the subspace, it fine-tunes the model for that specific goal

    This approach combines the efficiency of goal-conditioned RL with the precision of
    goal-specific fine-tuning when needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.env_prop.gc_adaptable()
            and not self.env_prop.is_state_discrete()
            and not self.env_prop.is_action_discrete()
        )
        if self.rl_agent_type is None:
            self.rl_agent_type = GCDeepRLAgent
        self.evaluation_function = kwargs.get("evaluation_function")
        if self.evaluation_function is None:
            from gr_libs.metrics.metrics import mean_wasserstein_distance

            self.evaluation_function = mean_wasserstein_distance
        assert callable(
            self.evaluation_function
        ), "Evaluation function must be a callable function."

        # Store fine-tuning parameters
        self.finetune_timesteps = kwargs.get("finetune_timesteps", FINETUNE_TIMESTEPS)

        # Dictionary to store fine-tuned agents for specific goals
        self.fine_tuned_agents = {}

    def domain_learning_phase(self, problems):
        base = problems["gc"]
        train_configs = base["train_configs"]

        # Store the goal subspace for later checks
        self.original_train_configs = train_configs

        super().domain_learning_phase(train_configs)

        agent_kwargs = {
            "domain_name": self.env_prop.domain_name,
            "problem_name": self.env_prop.name,
            "algorithm": train_configs[0][0],
            "num_timesteps": train_configs[0][1],
            "env_prop": self.env_prop,
        }

        agent = self.rl_agent_type(**agent_kwargs)
        agent.learn()
        self.agents[self.env_prop.name] = agent
        self.action_space = agent.env.action_space

    def _is_goal_in_subspace(self, goal):
        """
        Check if a goal is within the original training subspace.

        Delegates to the environment property's implementation.

        Args:
            goal: The goal to check

        Returns:
            bool: True if the goal is within the training subspace
        """
        # Use the environment property's implementation
        return self.env_prop.is_goal_in_subspace(goal)

    def goals_adaptation_phase(self, dynamic_goals):
        """
        Adapt to new goals, fine-tuning if necessary.

        For goals outside the original training subspace, fine-tune the model.

        Args:
            dynamic_goals: List of goals to adapt to
        """
        self.active_goals = dynamic_goals
        self.active_problems = [
            self.env_prop.goal_to_problem_str(goal) for goal in dynamic_goals
        ]

        # Check each goal and fine-tune if needed
        for goal in dynamic_goals:
            if not self._is_goal_in_subspace(goal):
                print(f"Goal {goal} is outside the training subspace. Fine-tuning...")

                # Create a new agent for this goal
                agent_kwargs = {
                    "domain_name": self.env_prop.domain_name,
                    "problem_name": self.env_prop.name,
                    "algorithm": self.original_train_configs[0][0],
                    "num_timesteps": self.original_train_configs[0][1],
                    "env_prop": self.env_prop,
                }

                # Create new agent with base model
                fine_tuned_agent = self.rl_agent_type(**agent_kwargs)
                fine_tuned_agent.learn()  # This loads the existing model

                # Fine-tune for this specific goal
                fine_tuned_agent.fine_tune(
                    goal=goal,
                    num_timesteps=self.finetune_timesteps,
                )

                # Store the fine-tuned agent
                self.fine_tuned_agents[
                    f"{self.env_prop.goal_to_str(goal)}_{self.finetune_timesteps}"
                ] = fine_tuned_agent
            else:
                print(f"Goal {goal} is within the training subspace. Using base agent.")

    def choose_agent(self, problem_name: str) -> RLAgent:
        """
        Return the appropriate agent for the given problem.

        If the goal has a fine-tuned agent, return that; otherwise return the base agent.

        Args:
            problem_name: The problem name to get agent for

        Returns:
            The appropriate agent (base or fine-tuned)
        """
        # Extract the goal from the problem name
        goal = self.env_prop.str_to_goal(problem_name)
        agent_name = f"{self.env_prop.goal_to_str(goal)}_{self.finetune_timesteps}"

        # Check if we have a fine-tuned agent for this goal
        if agent_name in self.fine_tuned_agents:
            return self.fine_tuned_agents[agent_name]

        # Otherwise return the base agent
        return self.agents[self.env_prop.name]
