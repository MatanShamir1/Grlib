""" metrics for GR algorithms, to perform distance, similarity, likelihood and other measurements and metrics. """

import math
from collections.abc import Callable, Generator
from math import log2
from typing import Any

import dill
import gymnasium
import numpy as np
from gymnasium.spaces.discrete import Discrete
from scipy.stats import wasserstein_distance

from ..ml.base import State
from ..ml.base.rl_agent import RLAgent
from ..ml.neural.deep_rl_learner import DeepRLAgent


def kl_divergence(p1: list[float], p2: list[float]) -> float:
    """
    Computes Kullback–Leibler divergence from two probabilities distributions p1 and p2.
    We follow the formula in Wikipedia https://en.wikipedia.org/wiki/Kullback–Leibler_divergence

    Args:
        p1 (List[float]): A probability distribution
        p2 (List[float]): Another probability distribution

    Returns:
        float: The KL-divergence between p1 and p2
    """
    assert len(p1) == len(p2)
    return sum(p1[i] * log2(p1[i] / p2[i]) for i in range(len(p1)))


def kl_divergence_norm_softmax(
    observations: list[tuple[State, Any]], agent, actions: Discrete
):
    """
    Calculates the Kullback-Leibler (KL) divergence between two probability distributions.

    Args:
        observations (list[tuple[State, Any]]): List of observations and corresponding actions.
        agent: The agent object.
        actions: The discrete actions.

    Returns:
        float: The mean KL divergence between the two distributions.
    """
    distances = []
    p_traj = traj_to_policy(observations=observations, actions=actions)

    for (observation, agent_pos), action in observations:
        state = observation["image"]
        state_pickled = dill.dumps(state)

        qp1 = p_traj[state_pickled]
        qp2_flatten_distribution_list: list[float] = agent.get_actions_probabilities(
            observation=(observation, agent_pos)
        )
        distances.append(kl_divergence(qp1, qp2_flatten_distribution_list))
    return np.mean(distances)


def amplify(values, alpha=1.0):
    """Computes amplified softmax probabilities for an array of values
    Args:
        values (list): Input values for which to compute softmax
        alpha (float): Amplification factor, where alpha > 1 increases differences between probabilities
    Returns:
        np.array: amplified softmax probabilities
    """
    values = values[:3] ** alpha  # currently only choose to turn or move forward
    return values / np.sum(values)


def stochastic_amplified_selection(actions_probs, alpha=8.0):
    """
    Selects an action based on the given action probabilities, with amplification using the specified alpha value.

    Parameters:
        actions_probs (list): A list of action probabilities.
        alpha (float): Amplification factor (default: 8.0).

    Returns:
        int: The selected action.

    """
    action_probs_amplified = amplify(actions_probs, alpha)
    choice = np.random.choice(len(action_probs_amplified), p=action_probs_amplified)
    if choice == 3:
        choice = 6
    return choice


import numpy as np


def stochastic_selection(actions_probs):
    """
    Selects an action based on the given probabilities using a stochastic selection method.

    Parameters:
        actions_probs (list): A list of probabilities for each action.

    Returns:
        int: The index of the selected action.
    """
    return np.random.choice(len(actions_probs), p=actions_probs)


def greedy_selection(actions_probs):
    """
    Selects the action with the highest probability.

    Args:
        actions_probs (numpy.ndarray): Array of action probabilities.

    Returns:
        int: Index of the selected action.
    """
    return np.argmax(actions_probs)


def measure_average_sequence_distance(seq1, seq2):
    """Measures the sequence similarity between two sequences of observations and actions.

    Args:
    seq1: A tensor of tensors representing the first sequence.
    seq2: A tensor of tensors representing the second sequence.

    Returns:
    A float representing the sequence similarity.
    """

    # Ensure both sequences have the same length
    min_seq_len = np.min([len(seq1), len(seq2)])
    assert (
        np.max([len(seq1), len(seq2)]) <= 30 * min_seq_len
    ), "We can't really measure similarity in case the sequences are really not the same... maybe just return a default NOT_SIMILAR here."

    # Calculate the Euclidean distance between corresponding elements in the sequences
    distances = []
    for i in range(0, min_seq_len):
        distances.append(np.sum(np.abs(np.array(seq1[i]) - np.array(seq2[i]))))

    # Calculate the average distance over all elements
    return np.mean(np.array(distances))


def traj_to_policy(
    observations: list[tuple[State, Any]], actions: Discrete, epsilon: float = 0.0
) -> dict[str, list[float]]:
    """
    Converts a trajectory from a planner to a policy.

    Args:
        observations (list[tuple[State, Any]]): List of tuples containing the observation and the corresponding action.
        actions (Discrete): Discrete action space.
        epsilon (float, optional): Exploration parameter. Defaults to 0.0.

    Returns:
        dict[str, list[float]]: Dictionary mapping serialized states to action probabilities.
    """
    trajectory_as_policy = {}
    for (observation, _agent_pos), action in observations:
        action_index = action

        actions_len = actions.n
        qs = [1e-6 + epsilon / actions_len for _ in range(actions_len)]
        qs[action_index] = 1.0 - 1e-6 * (actions_len - 1) - epsilon

        state = observation["image"]
        state_pickled = dill.dumps(state)
        trajectory_as_policy[state_pickled] = qs
    return trajectory_as_policy


from collections.abc import Generator
from typing import Any


def pass_observation_patcher(
    observations: list[Any], agent: RLAgent
) -> Generator[None, None, None]:
    """
    Generator function that yields observations.

    Args:
        observations (list): List of observations.
        agent (RLAgent): RL agent object.

    Yields:
        None: Yields each observation from the list.

    """
    yield from observations


def mean_wasserstein_distance(
    observations: list[tuple[State, Any]],
    agent: DeepRLAgent,
    actions: gymnasium.spaces.Box,
    observation_patcher: Callable[
        [list[Any], RLAgent], Generator[None, None, None]
    ] = pass_observation_patcher,
):
    """
    Calculates the mean Wasserstein distance between observed actions and actor means.

    Args:
        observations (list[tuple[State, Any]]): List of observations and corresponding actions.
        agent (DeepRLAgent): The deep reinforcement learning agent.
        actions (gymnasium.spaces.Box): The action space.
        observation_patcher (Callable[[list[Any], RLAgent], Generator[None, None, None]], optional):
            A function that patches the observations. Defaults to pass_observation_patcher.

    Returns:
        float: The mean Wasserstein distance between observed actions and actor means.
    """
    distances = []

    for observation, observed_action in observation_patcher(observations, agent):
        # execute prediction X times and add to list (observed_action * X) |X| Len
        actor_means, log_std_dev = agent.get_mean_and_std_dev(observation=observation)

        # split to 3 axis and for each one calculate wasserstein distance and report mean
        observed_action = observed_action[0]
        actor_means = actor_means[0]

        if len(observed_action) != len(actor_means):
            raise Exception(
                f"Length of observed actions, actor mean should be equal! "
                f"{len(observed_action)},{len(actor_means)}"
            )
        wasserstein_distances = []
        for observation_action, actor_mean in zip(observed_action, actor_means):
            wasserstein_distances.append(
                wasserstein_distance([observation_action], [actor_mean])
            )
        distances.append(np.mean(wasserstein_distances))
    return np.mean(distances)


def mean_action_distance_continuous(
    observations: list[tuple[State, Any]],
    agent: DeepRLAgent,
    actions: gymnasium.spaces.Box,
):
    """
    Calculates the mean distance between the predicted actions and the actual actions for a continuous action space.

    Args:
        observations (list[tuple[State, Any]]): A list of tuples containing the observations and corresponding actions.
        agent (DeepRLAgent): The deep reinforcement learning agent used to predict actions.
        actions (gymnasium.spaces.Box): The action space.

    Returns:
        float: The mean distance between the predicted actions and the actual actions.
    """
    distances = []
    for observation, action in observations:
        action2, _ = agent.model.predict(
            observation,
            state=None,
            deterministic=True,
            episode_start=np.ones((1,), dtype=bool),
        )
        action_arr, action2_arr = action[0], action2[0]
        print(f"actor means:{action2}")
        assert len(action_arr) == len(
            action2_arr
        ), f"Actions should be on the same length:{action},{action2}"

        total_diff = 0
        for action1, action2 in zip(action_arr, action2_arr):
            total_diff += math.fabs(action1 - action2)
        distances.append(total_diff)
    return np.mean(distances)


from collections.abc import Generator
from typing import Any


def set_agent_goal_observation(
    observations: list[Any], agent: RLAgent
) -> Generator[None, None, None]:
    """
    Sets the desired goal in each observation to the agent's goal.

    Args:
        observations (list): List of observations.
        agent (RLAgent): The RL agent.

    Yields:
        tuple: A tuple containing the modified observation and the corresponding action.
    """
    copy_observation = observations.copy()
    for observation, action in copy_observation:
        observation["desired_goal"] = agent.goal
        yield observation, action


def z_score(x, mean_action: float, std_dev: float):
    return (x - mean_action) / std_dev


def mean_p_value(
    observations: list[tuple[State, Any]],
    agent: DeepRLAgent,
    actions: gymnasium.spaces.Box,
    observation_patcher: Callable[
        [list[Any], RLAgent], Generator[None, None, None]
    ] = pass_observation_patcher,
):
    """
    Calculate the mean p-value for a given set of observations.

    Args:
        observations (list[tuple[State, Any]]): List of observations and corresponding actions.
        agent (DeepRLAgent): The deep reinforcement learning agent.
        actions (gymnasium.spaces.Box): The action space.
        observation_patcher (Callable[[list[Any], RLAgent], Generator[None, None, None]], optional):
            A function that patches the observations. Defaults to pass_observation_patcher.

    Returns:
        float: The mean p-value.

    Raises:
        Exception: If the lengths of observed actions, actor mean, and std-dev are not equal.
    """
    distances = []
    for observation, observed_action in observation_patcher(observations, agent):
        # execute prediction X times and add to list (observed_action * X) |X| Len
        actor_means, log_std_dev = agent.get_mean_and_std_dev(observation=observation)

        # for each axis, calculate z-score distance and report mean
        actor_means = actor_means[0]
        observed_actions = observed_action[0]
        log_std_dev = log_std_dev[0]

        if (
            len(actor_means) != len(observed_actions)
            or len(actor_means) != len(log_std_dev)
            or len(observed_actions) != len(log_std_dev)
        ):
            raise Exception(
                f"Length of observed actions, actor mean and std-dev should be equal! "
                f"{len(observed_actions)},{len(actor_means)},{len(log_std_dev)}"
            )
        z_scores = []
        for actor_mean, observation_action, action_log_std_dev in zip(
            actor_means, observed_actions, log_std_dev
        ):
            z_scores.append(
                math.fabs(
                    z_score(
                        observation_action,
                        actor_mean,
                        math.pow(2, math.fabs(action_log_std_dev)),
                    )
                )
            )
        mean_distances = np.mean(z_scores)

        distances.append(mean_distances)
    return np.mean(distances)


def normalize(values: list[float]) -> list[float]:
    """
    Normalize a list of values by dividing each value by the sum of all values.

    Args:
        values (list[float]): The list of values to be normalized.

    Returns:
        list[float]: The normalized list of values.
    """
    values /= sum(values)
    return values


def maximum(values: list[float]) -> list[float]:
    """
    Returns a list with the same length as the input list, where the maximum value is set to 1.0 and all other values are set to 0.0.

    Args:
        values (list[float]): The input list of values.

    Returns:
        list[float]: A list with the same length as the input list, where the maximum value is set to 1.0 and all other values are set to 0.0.
    """
    if not len(values):
        return values
    vals = np.array(values)
    argmax = vals.argmax()
    vals[:] = 0.0
    vals[argmax] = 1.0
    return vals
