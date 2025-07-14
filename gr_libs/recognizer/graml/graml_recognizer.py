"""Collection of recognizers that use GRAML methods: metric learning for ODGR."""

import os
from abc import abstractmethod

import dill
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from gr_libs.environment.environment import EnvProperty
from gr_libs.metrics import metrics
from gr_libs.ml import utils
from gr_libs.ml.base import ContextualAgent
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent, GCDeepRLAgent
from gr_libs.ml.planner.mcts import mcts_model
from gr_libs.ml.sequential._lstm_model import LstmObservations, train_metric_model
from gr_libs.ml.tabular.tabular_q_learner import TabularQLearner
from gr_libs.ml.utils.format import random_subset_with_order
from gr_libs.ml.utils.storage import (
    get_and_create,
    get_embeddings_result_path,
    get_lstm_model_dir,
    get_policy_sequences_result_path,
)
from gr_libs.recognizer.graml._gr_dataset import GRDataset, generate_datasets
from gr_libs.recognizer.recognizer import (
    GaAdaptingRecognizer,
    GaAgentTrainerRecognizer,
    LearningRecognizer,
)

### TODO IMPLEMENT MORE SELECTION METHODS, MAKE SURE action_probs IS AS IT SEEMS: list of action-probability 'es ###


def collate_fn(batch):
    """
    Collates a batch of data for training or evaluation.

    Args:
        batch (list): A list of tuples, where each tuple contains the first traces, second traces, and the label indicating whether the goals are the same.

    Returns:
        tuple: A tuple containing the padded first traces, padded second traces, labels, lengths of first traces, and lengths of second traces.
    """
    first_traces, second_traces, is_same_goals = zip(*batch)
    # torch.stack takes tensor tuples (fixed size) and stacks them up in a matrix
    first_traces_padded = pad_sequence(
        [torch.stack(sequence) for sequence in first_traces], batch_first=True
    )
    second_traces_padded = pad_sequence(
        [torch.stack(sequence) for sequence in second_traces], batch_first=True
    )
    first_traces_lengths = [len(trace) for trace in first_traces]
    second_traces_lengths = [len(trace) for trace in second_traces]
    return (
        first_traces_padded.to(utils.device),
        second_traces_padded.to(utils.device),
        torch.stack(is_same_goals).to(utils.device),
        first_traces_lengths,
        second_traces_lengths,
    )


def load_weights(loaded_model: LstmObservations, path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model.load_state_dict(torch.load(path, map_location=utils.device))
    loaded_model.to(utils.device)  # Ensure model is on the right device
    return loaded_model


def save_weights(model: LstmObservations, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), path)


class Graml(LearningRecognizer):
    """
    The Graml class is a subclass of LearningRecognizer and represents a recognizer that uses the Graml algorithm for goal recognition.
    Graml learns a metric over observation sequences, over time: using a GC or a collection of agents, it creates a dataset and learns
    the metric on it during the domain learning phase. During the goals adaptation phase, it creates or receives a library of sequences for each goal,
    and maintains embeddings of them for the inference phase. The inference phase uses the learned metric to find the closest goal to a given sequence.

    Attributes:
        agents (list[ContextualAgent]): A list of contextual agents associated with the recognizer.
        train_func: The function used for training the metric model.
        collate_func: The function used for collating data in the training process.

    Methods:
        train_agents_on_base_goals(base_goals: list[str], train_configs: list): Trains the agents on the given base goals and train configurations.
        domain_learning_phase(base_goals: list[str], train_configs: list): Performs the domain learning phase of the Graml algorithm.
        goals_adaptation_phase(dynamic_goals: list[EnvProperty], save_fig=False): Performs the goals adaptation phase of the Graml algorithm.
        get_goal_plan(goal): Retrieves the plan associated with the given goal.
        dump_plans(true_sequence, true_goal, percentage): Dumps the plans to a file.
        create_embeddings_dict(): Creates the embeddings dictionary for the plans.
        inference_phase(inf_sequence, true_goal, percentage) -> str: Performs the inference phase of the Graml algorithm and returns the closest goal.
        generate_sequences_library(goal: str, save_fig=False) -> list[list[tuple[np.ndarray, np.ndarray]]]: Generates the sequences library for the given goal.
        update_sequences_library_inference_phase(inf_sequence) -> list[list[tuple[np.ndarray, np.ndarray]]]: Updates the sequences library during the inference phase.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the GramlRecognizer object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Attributes:
            agents (list[ContextualAgent]): List of contextual agents.
            train_func: Training function for the metric model.
            collate_func: Collate function for data batching.
        """
        super().__init__(*args, **kwargs)
        self.agents: list[ContextualAgent] = []
        self.train_func = train_metric_model
        self.collate_func = collate_fn

    @abstractmethod
    def train_agents_on_base_goals(self, base_goals: list[str], train_configs: list):
        pass

    def domain_learning_phase(self, base_goals: list[str], train_configs: list):
        super().domain_learning_phase(train_configs, base_goals)
        self.train_agents_on_base_goals(base_goals, train_configs)
        # train the network so it will find a metric for the observations of the base agents such that traces of agents to different goals are far from one another
        self.model_directory = get_lstm_model_dir(
            domain_name=self.env_prop.domain_name,
            env_name=self.env_prop.name,
            model_name=self.env_prop.problem_list_to_str_tuple(self.original_problems),
            recognizer=self.__class__.__name__,
        )
        last_path = r"lstm_model.pth"
        self.model_file_path = os.path.join(self.model_directory, last_path)
        self.model = LstmObservations(
            input_size=self.env_prop.get_lstm_props().input_size,
            hidden_size=self.env_prop.get_lstm_props().hidden_size,
        )
        self.model.to(utils.device)

        if os.path.exists(self.model_file_path):
            print(f"Loading pre-existing lstm model in {self.model_file_path}")
            load_weights(loaded_model=self.model, path=self.model_file_path)
        else:
            print(f"{self.model_file_path} doesn't exist, training the model")
            train_samples, dev_samples = generate_datasets(
                num_samples=self.env_prop.get_lstm_props().num_samples,
                agents=self.agents,
                observation_creation_method=metrics.stochastic_amplified_selection,
                problems=self.original_problems,
                env_prop=self.env_prop,
                gc_goal_set=self.gc_goal_set if hasattr(self, "gc_goal_set") else None,
                recognizer_name=self.__class__.__name__,
            )

            train_dataset = GRDataset(len(train_samples), train_samples)
            dev_dataset = GRDataset(len(dev_samples), dev_samples)
            self.train_func(
                self.model,
                train_loader=DataLoader(
                    train_dataset,
                    batch_size=self.env_prop.get_lstm_props().batch_size,
                    shuffle=False,
                    collate_fn=self.collate_func,
                ),
                dev_loader=DataLoader(
                    dev_dataset,
                    batch_size=self.env_prop.get_lstm_props().batch_size,
                    shuffle=False,
                    collate_fn=self.collate_func,
                ),
            )
            save_weights(model=self.model, path=self.model_file_path)

    def goals_adaptation_phase(self, dynamic_goals: list[EnvProperty], save_fig=False):
        self.is_first_inf_since_new_goals = True
        self.current_goals = dynamic_goals
        # start by training each rl agent on the base goal set
        self.embeddings_dict = (
            {}
        )  # relevant if the embedding of the plan occurs during the goals adaptation phase
        self.plans_dict = (
            {}
        )  # relevant if the embedding of the plan occurs during the inference phase
        for goal in self.current_goals:
            obss = self.generate_sequences_library(goal, save_fig=save_fig)
            self.plans_dict[str(goal)] = obss

    def get_goal_plan(self, goal):
        assert (
            self.plans_dict
        ), "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't return the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
        return self.plans_dict[goal]

    def dump_plans(self, true_sequence, true_goal, percentage):
        assert (
            self.plans_dict
        ), "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't return the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
        # Arrange storage
        embeddings_path = get_and_create(
            get_embeddings_result_path(
                domain_name=self.env_prop.domain_name,
                env_name=self.env_prop.name,
                recognizer=self.__class__.__name__,
            )
        )
        self.plans_dict[f"{true_goal}_true"] = true_sequence

        with open(
            embeddings_path + f"/{true_goal}_{percentage}_plans_dict.pkl", "wb"
        ) as plans_file:
            to_dump = {}
            for goal, obss in self.plans_dict.items():
                if goal == f"{true_goal}_true":
                    to_dump[goal] = self.agents[0].agent.simplify_observation(obss)
                else:
                    to_dump[goal] = []
                    for obs in obss:
                        addition = (
                            self.agents[0].agent.simplify_observation(obs)
                            if self.is_first_inf_since_new_goals
                            else obs
                        )
                        to_dump[goal].append(addition)
            dill.dump(to_dump, plans_file)
        self.plans_dict.pop(f"{true_goal}_true")

    def create_embeddings_dict(self):
        for goal, obss in self.plans_dict.items():
            self.embeddings_dict[goal] = []
            for cons_seq, non_cons_seq in obss:
                self.embeddings_dict[goal].append(
                    (
                        self.model.embed_sequence(cons_seq),
                        self.model.embed_sequence(non_cons_seq),
                    )
                )

    def inference_phase(self, inf_sequence, true_goal, percentage) -> str:
        embeddings_path = get_and_create(
            get_embeddings_result_path(
                domain_name=self.env_prop.domain_name,
                env_name=self.env_prop.name,
                recognizer=self.__class__.__name__,
            )
        )
        simplified_inf_sequence = self.agents[0].agent.simplify_observation(
            inf_sequence
        )
        new_embedding = self.model.embed_sequence(simplified_inf_sequence)
        assert (
            self.plans_dict
        ), "plans_dict wasn't created during goals_adaptation_phase and now inference phase can't embed the plans. when inference_same_length, keep the plans and not their embeddings during goals_adaptation_phase."
        if self.is_first_inf_since_new_goals:
            self.is_first_inf_since_new_goals = False
            self.update_sequences_library_inference_phase(inf_sequence)
            self.create_embeddings_dict()

        closest_goal, greatest_similarity = None, 0
        for goal, embeddings in self.embeddings_dict.items():
            sum_curr_similarities = 0
            for cons_embedding, non_cons_embedding in embeddings:
                sum_curr_similarities += max(
                    torch.exp(-torch.sum(torch.abs(cons_embedding - new_embedding))),
                    torch.exp(
                        -torch.sum(torch.abs(non_cons_embedding - new_embedding))
                    ),
                )
            mean_similarity = sum_curr_similarities / len(embeddings)
            if mean_similarity > greatest_similarity:
                closest_goal = goal
                greatest_similarity = mean_similarity

        self.embeddings_dict[f"{true_goal}_true"] = new_embedding
        if self.collect_statistics:
            with open(
                os.path.join(
                    embeddings_path, f"{true_goal}_{percentage}_embeddings_dict.pkl"
                ),
                "wb",
            ) as embeddings_file:
                dill.dump(self.embeddings_dict, embeddings_file)
        self.embeddings_dict.pop(f"{true_goal}_true")

        return closest_goal

    @abstractmethod
    def generate_sequences_library(
        self, goal: str, save_fig=False
    ) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        pass

    # this function duplicates every sequence and creates a consecutive and non-consecutive version of it
    def update_sequences_library_inference_phase(
        self, inf_sequence
    ) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        new_plans_dict = {}
        for goal, obss in self.plans_dict.items():
            new_obss = []
            for obs in obss:
                consecutive_partial_obs = random_subset_with_order(
                    obs, len(inf_sequence), is_consecutive=True
                )
                non_consecutive_partial_obs = random_subset_with_order(
                    obs, len(inf_sequence), is_consecutive=False
                )
                simplified_consecutive_partial_obs = self.agents[
                    0
                ].agent.simplify_observation(consecutive_partial_obs)
                simplified_non_consecutive_partial_obs = self.agents[
                    0
                ].agent.simplify_observation(non_consecutive_partial_obs)
                new_obss.append(
                    (
                        simplified_consecutive_partial_obs,
                        simplified_non_consecutive_partial_obs,
                    )
                )
            new_plans_dict[goal] = (
                new_obss  # override old full observations with new partial observations with consecutive and non-consecutive versions.
            )
        self.plans_dict = new_plans_dict


class BGGraml(Graml):
    """
    BGGraml class represents a goal-directed agent for the BGGraml algorithm.

    It extends the Graml class and provides additional methods for training agents on base goals.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def domain_learning_phase(self, problems):
        # Always use 'bg' for BGGraml
        base = problems["bg"]
        base_goals = base["goals"]
        train_configs = base["train_configs"]
        assert len(base_goals) == len(
            train_configs
        ), "base_goals and train_configs should have the same length"
        super().domain_learning_phase(
            train_configs=train_configs, base_goals=base_goals
        )

    # In case we need goal-directed agent for every goal
    def train_agents_on_base_goals(self, base_goals: list[str], train_configs: list):
        self.original_problems = [
            self.env_prop.goal_to_problem_str(g) for g in base_goals
        ]
        # start by training each rl agent on the base goal set
        for (problem, goal), (algorithm, num_timesteps) in zip(
            zip(self.original_problems, base_goals), train_configs
        ):
            kwargs = {
                "domain_name": self.domain_name,
                "problem_name": problem,
                "env_prop": self.env_prop,
            }
            if algorithm != None:
                kwargs["algorithm"] = algorithm
            if num_timesteps != None:
                kwargs["num_timesteps"] = num_timesteps
            agent = self.rl_agent_type(**kwargs)
            agent.learn()
            self.agents.append(
                ContextualAgent(problem_name=problem, problem_goal=goal, agent=agent)
            )


class MCTSBasedGraml(BGGraml, GaAdaptingRecognizer):
    """
    MCTSBasedGraml is a class that represents a recognizer based on the MCTS algorithm.
    It inherits from BGGraml and GaAdaptingRecognizer classes.

    Attributes:
        rl_agent_type (type): The type of reinforcement learning agent used.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the GramlRecognizer object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        super().__init__(*args, **kwargs)
        if self.rl_agent_type == None:
            self.rl_agent_type = TabularQLearner

    def generate_sequences_library(
        self, goal: str, save_fig=False
    ) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        """
        Generates a library of sequences for a given goal.

        Args:
            goal (str): The goal for which to generate sequences.
            save_fig (bool, optional): Whether to save the generated figure. Defaults to False.

        Returns:
            list[list[tuple[np.ndarray, np.ndarray]]]: The generated sequences library.
        """
        problem_name = self.env_prop.goal_to_problem_str(goal)
        img_path = (
            os.path.join(
                get_policy_sequences_result_path(
                    domain_name=self.env_prop.domain_name,
                    env_name=self.env_prop.name,
                    recognizer=self.__class__.__name__,
                ),
                problem_name + "_MCTS",
            )
            if save_fig
            else None
        )
        return [
            mcts_model.plan(
                self.env_prop.domain_name,
                problem_name,
                goal,
                save_fig=save_fig,
                fig_path=img_path,
                env_prop=self.env_prop,
            )
        ]


class ExpertBasedGraml(BGGraml, GaAgentTrainerRecognizer):
    """
    ExpertBasedGraml class represents a Graml recognizer that uses expert knowledge to generate sequences library and adapt goals.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        rl_agent_type (type): The type of reinforcement learning agent used.
        env_prop (EnvironmentProperties): The environment properties.
        dynamic_train_configs_dict (dict): The dynamic training configurations for each problem.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the GRAML Recognizer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        super().__init__(*args, **kwargs)
        if self.rl_agent_type == None:
            if self.env_prop.is_state_discrete() and self.env_prop.is_action_discrete():
                self.rl_agent_type = TabularQLearner
            else:
                self.rl_agent_type = DeepRLAgent

    def generate_sequences_library(
        self, goal: str, save_fig=False
    ) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        """
        Generates a sequences library for a given goal.

        Args:
            goal (str): The goal for which to generate the sequences library.
            save_fig (bool, optional): Whether to save the figure. Defaults to False.

        Returns:
            list[list[tuple[np.ndarray, np.ndarray]]]: The generated sequences library.

        """
        problem_name = self.env_prop.goal_to_problem_str(goal)
        kwargs = {
            "domain_name": self.domain_name,
            "problem_name": problem_name,
            "env_prop": self.env_prop,
        }
        if self.dynamic_train_configs_dict[problem_name][0] != None:
            kwargs["algorithm"] = self.dynamic_train_configs_dict[problem_name][0]
        if self.dynamic_train_configs_dict[problem_name][1] != None:
            kwargs["num_timesteps"] = self.dynamic_train_configs_dict[problem_name][1]
        agent = self.rl_agent_type(**kwargs)
        agent.learn()
        agent_kwargs = {
            "action_selection_method": metrics.greedy_selection,
            "random_optimalism": False,
            "save_fig": save_fig,
        }
        if save_fig:
            fig_path = get_and_create(
                f"{os.path.abspath(os.path.join(get_policy_sequences_result_path(domain_name=self.env_prop.domain_name, env_name=self.env_prop.name, recognizer=self.__class__.__name__), problem_name))}_bg_sequence"
            )
            agent_kwargs["fig_path"] = fig_path
        return [agent.generate_observation(**agent_kwargs)]

    def goals_adaptation_phase(self, dynamic_goals: list[str], dynamic_train_configs):
        """
        Performs the goals adaptation phase.

        Args:
            dynamic_goals (list[str]): The dynamic goals.
            dynamic_train_configs: The dynamic training configurations.

        Returns:
            The result of the goals adaptation phase.

        """
        assert len(dynamic_goals) == len(
            dynamic_train_configs
        ), "dynamic_goals and dynamic_train_configs should have the same length"
        self.dynamic_goals_problems = [
            self.env_prop.goal_to_problem_str(g) for g in dynamic_goals
        ]
        self.dynamic_train_configs_dict = {
            problem: config
            for problem, config in zip(
                self.dynamic_goals_problems, dynamic_train_configs
            )
        }
        return super().goals_adaptation_phase(dynamic_goals)


class GCGraml(Graml, GaAdaptingRecognizer):
    """
    GCGraml class represents a recognizer that uses the GCDeepRLAgent for domain learning and sequence generation.
    It makes its adaptation phase quicker and require less assumptions, but the assumption of a GC agent is still needed and may result
    in less optimal policies that generate the observations in the synthetic dataset, which could eventually lead to a less optimal metric.

    Args:
        Graml (class): Base class for Graml recognizers.
        GaAdaptingRecognizer (class): Base class for GA adapting recognizers.

    Attributes:
        rl_agent_type (class): The type of RL agent to be used for learning and generation.
        env_prop (object): The environment properties.
        agents (list): List of contextual agents.

    Methods:
        __init__: Initializes the GCGraml recognizer.
        domain_learning_phase: Performs the domain learning phase.
        train_agents_on_base_goals: Trains the RL agents on the base goals.
        generate_sequences_library: Generates sequences library for a specific goal.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rl_agent_type == None:
            self.rl_agent_type = GCDeepRLAgent
        assert (
            self.env_prop.gc_adaptable()
            and not self.env_prop.is_state_discrete()
            and not self.env_prop.is_action_discrete()
        )

    def domain_learning_phase(self, problems):
        # Always use 'gc' for GCGraml
        base = problems["gc"]
        base_goals = base["goals"]
        train_configs = base["train_configs"]
        assert (
            len(train_configs) == 1
        ), "GCGraml should only have one train config for the base goals, it uses a single agent"
        super().domain_learning_phase(
            train_configs=train_configs, base_goals=base_goals
        )

    # In case we need goal-directed agent for every goal
    def train_agents_on_base_goals(self, base_goals: list[str], train_configs: list):
        self.gc_goal_set = base_goals
        self.original_problems = self.env_prop.name  # needed for gr_dataset
        # start by training each rl agent on the base goal set
        kwargs = {
            "domain_name": self.domain_name,
            "problem_name": self.env_prop.name,
            "env_prop": self.env_prop,
        }
        algorithm, num_timesteps = train_configs[0]  # should only be one, was asserted
        if algorithm != None:
            kwargs["algorithm"] = algorithm
        if num_timesteps != None:
            kwargs["num_timesteps"] = num_timesteps
        gc_agent = self.rl_agent_type(**kwargs)
        gc_agent.learn()
        self.agents.append(
            ContextualAgent(
                problem_name=self.env_prop.name, problem_goal="general", agent=gc_agent
            )
        )

    def generate_sequences_library(
        self, goal: str, save_fig=False
    ) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        problem_name = self.env_prop.goal_to_problem_str(goal)
        kwargs = {
            "domain_name": self.domain_name,
            "problem_name": self.env_prop.name,
            "env_prop": self.env_prop,
        }  # problem name is env name in gc case
        if self.original_train_configs[0][0] != None:
            kwargs["algorithm"] = self.original_train_configs[0][0]
        if self.original_train_configs[0][1] != None:
            kwargs["num_timesteps"] = self.original_train_configs[0][1]
        agent = self.rl_agent_type(**kwargs)
        agent.learn()
        agent_kwargs = {
            "action_selection_method": metrics.stochastic_amplified_selection,
            "random_optimalism": True,
            "save_fig": save_fig,
        }
        if save_fig:
            fig_path = get_and_create(
                f"{os.path.abspath(os.path.join(get_policy_sequences_result_path(domain_name=self.env_prop.domain_name, env_name=self.env_prop.name, recognizer=self.__class__.__name__), problem_name))}_gc_sequence"
            )
            agent_kwargs["fig_path"] = fig_path
        if self.env_prop.use_goal_directed_problem():
            agent_kwargs["goal_directed_problem"] = problem_name
        else:
            agent_kwargs["goal_directed_goal"] = goal
        obss = []
        for _ in range(5):
            obss.append(agent.generate_observation(**agent_kwargs))
        return obss
