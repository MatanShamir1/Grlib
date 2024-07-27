import numpy
import re
import torch
import ml
import gymnasium as gym
import random


def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return ml.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])

        def preprocess_obss(obss, device=None):
            return ml.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })

        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss

def goal_to_minigrid_str(tuply):
	tuply = tuply[1:-1] # remove the braces
	#print(tuply)
	nums = tuply.split(',')
	#print(nums)
	return f'MiniGrid-SimpleCrossingS13N4-DynamicGoal-{nums[0]}x{nums[1]}-v0'

def minigrid_str_to_goal(str):
	"""
	This function extracts the goal size (width and height) from a MiniGrid environment name.

	Args:
		env_name: The name of the MiniGrid environment (string).

	Returns:
		A tuple of integers representing the goal size (width, height).
	"""
	# Split the environment name by separators
	parts = str.split("-")
	# Find the part containing the goal size (usually after "DynamicGoal")
	goal_part = [part for part in parts if "x" in part]
	# Extract width and height from the goal part
	width, height = goal_part[0].split("x")
	return (int(width), int(height))

def goal_str_to_tuple(str):
    assert str[0] == "(" and str[-1] == ")"
    str = str[1:-1]
    width, height = str.split(',')
    return (int(width), int(height))


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def random_subset_with_order(sequence, subset_size, is_fragmented = True):
    if subset_size >= len(sequence):
        return sequence
    else:
        if is_fragmented:
            indices_to_select = sorted(random.sample(range(len(sequence)), subset_size))  # Randomly select indices to keep
        else:
            indices_to_select = [i for i in range(subset_size)]
        return [sequence[i] for i in indices_to_select]  # Return the elements corresponding to the selected indices



def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
