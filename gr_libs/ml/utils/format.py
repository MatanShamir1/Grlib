""" formatting-related utilities """

import random


def random_subset_with_order(sequence, subset_size, is_consecutive=True):
    """
    Returns a random subset of elements from the given sequence with a specified subset size.

    Args:
        sequence (list): The sequence of elements to select from.
        subset_size (int): The size of the desired subset.
        is_consecutive (bool, optional): Whether the selected subset should be consecutive elements from the sequence.
            Defaults to True.

    Returns:
        list: A random subset of elements from the sequence.

    """
    if not isinstance(sequence, list):
        raise TypeError(f"Expected observation to be an iterable, got {type(sequence)}")
    if subset_size >= len(sequence):
        return sequence
    else:
        if is_consecutive:
            indices_to_select = [i for i in range(subset_size)]
        else:
            indices_to_select = sorted(
                random.sample(range(len(sequence)), subset_size)
            )  # Randomly select indices to keep
        return [
            sequence[i] for i in indices_to_select
        ]  # Return the elements corresponding to the selected indices
