import math
import dill
import numpy as np

from typing import List, Dict, Tuple, Any
from math import log2
from numpy.core.fromnumeric import mean
from gymnasium.spaces.discrete import Discrete
from torch.distributions.categorical import Categorical

# from ml.neural import BaseAlgo
from ml.base import State


def kl_divergence(p1: List[float], p2: List[float]) -> float:
    """Computes Kullback–Leibler divergence from two probabilities distributions p1 and p2.
    We follow the formula in Wikipedia https://en.wikipedia.org/wiki/Kullback–Leibler_divergence

    Args:
        p1 (List[float]): A probability distribution
        p2 (List[float]): Another probability distribution

    Returns:
        float: The KL-divergence between p1 and p2
    """
    assert (len(p1) == len(p2))
    return sum(p1[i] * log2(p1[i] / p2[i]) for i in range(len(p1)))


# def kl_divergence_norm_softmax(observations: List[Tuple[State, Any]], agent: BaseAlgo, actions: Discrete):
#     distances = []
#     p_traj = traj_to_policy(observations=observations, actions=actions)

#     for (observation, agent_pos), action in observations:
#         state = observation['image']
#         state_pickled = dill.dumps(state)

#         qp1 = p_traj[state_pickled]
#         qp2_flatten_distribution_list: List[float] = agent.get_actions_probabilities(
#             observation=(observation, agent_pos))
#         distances.append(kl_divergence(qp1, qp2_flatten_distribution_list))
#     return mean(distances)


def softmax(values: List[float]) -> List[float]:
    """Computes softmax probabilities for an array of values
    TODO We should probably use numpy arrays here
    Args:
        values (np.array): Input values for which to compute softmax

    Returns:
        np.array: softmax probabilities
    """
    return [(math.exp(q)) / sum([math.exp(_q) for _q in values]) for q in values]

def amplify(values, alpha=1.0):
    """Computes amplified softmax probabilities for an array of values
    Args:
        values (list): Input values for which to compute softmax
        alpha (float): Amplification factor, where alpha > 1 increases differences between probabilities
    Returns:
        np.array: amplified softmax probabilities
    """
    values = np.concatenate((values[:3], values[-1:]))**alpha # currently only choose to turn, move forward or finish
    return values / np.sum(values)

def stochastic_amplified_selection(actions_probs, alpha=15.0):
    action_probs_amplified = amplify(actions_probs, alpha)
    choice = np.random.choice(len(action_probs_amplified), p=action_probs_amplified)
    if choice == 3:
        choice = 6
    return choice

def stochastic_selection(actions_probs):
    return np.random.choice(len(actions_probs), p=actions_probs)

def greedy_selection(actions_probs):
    return np.argmax(actions_probs)


def traj_to_policy(observations: List[Tuple[State, Any]], actions: Discrete, epsilon: float = 0.) -> Dict[
    str, List[float]]:
    # converts a trajectory from a planner to a policy
    # where the taken action has 99.99999% probability
    trajectory_as_policy = {}
    for (observation, agent_pos), action in observations:
        # in the discrete world the action is the index
        action_index = action

        actions_len = actions.n
        qs = [1e-6 + epsilon / actions_len for _ in range(actions_len)]
        qs[action_index] = 1. - 1e-6 * (actions_len - 1) - epsilon

        state = observation['image']
        state_pickled = dill.dumps(state)
        trajectory_as_policy[state_pickled] = qs
    return trajectory_as_policy


def normalize(values: List[float]) -> List[float]:
    values /= sum(values)
    return values

def max(values: List[float]) -> List[float]:
    if not len(values):
        return values
    vals = np.array(values)
    argmax = vals.argmax()
    vals[:] = 0.0
    vals[argmax] = 1.0
    return vals