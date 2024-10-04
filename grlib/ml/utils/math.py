import math
from typing import Callable, Generator, List

def softmax(values: List[float]) -> List[float]:
    """Computes softmax probabilities for an array of values
    TODO We should probably use numpy arrays here
    Args:
        values (np.array): Input values for which to compute softmax

    Returns:
        np.array: softmax probabilities
    """
    return [(math.exp(q)) / sum([math.exp(_q) for _q in values]) for q in values]