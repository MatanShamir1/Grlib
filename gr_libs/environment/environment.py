""" environment.py """

import os
from abc import abstractmethod
from collections import namedtuple

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from minigrid.core.world_object import Lava, Wall
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image
from stable_baselines3.common.vec_env import DummyVecEnv

MINIGRID, PANDA, PARKING, POINT_MAZE = "minigrid", "panda", "parking", "point_maze"

QLEARNING = "QLEARNING"

SUPPORTED_DOMAINS = [MINIGRID, PANDA, PARKING, POINT_MAZE]

LSTMProperties = namedtuple(
    "LSTMProperties", ["input_size", "hidden_size", "batch_size", "num_samples"]
)


class EnvProperty:
    """
    Base class for environment properties.
    """

    def __init__(self, name):
        """
        Initializes a new instance of the Environment class.

        Args:
            name (str): The name of the environment.
        """
        self.name = name

    def __str__(self):
        """
        Returns a string representation of the object.
        """
        return f"{self.name}"

    def __repr__(self):
        """
        Returns a string representation of the object.
        """
        return f"{self.name}"

    def __eq__(self, other):
        """
        Check if this object is equal to another object.

        Args:
            other: The other object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        return self.name == other.name

    def __ne__(self, other):
        """
        Check if the current object is not equal to the other object.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the objects are not equal, False otherwise.
        """
        return not self.__eq__(other)

    @abstractmethod
    def str_to_goal(self):
        """
        Convert a problem name to a goal.
        """

    @abstractmethod
    def gc_adaptable(self):
        """
        Check if the environment is goal-conditioned adaptable.
        """

    @abstractmethod
    def problem_list_to_str_tuple(self, problems):
        """
        Convert a list of problems to a string tuple.
        """

    @abstractmethod
    def goal_to_problem_str(self, goal):
        """
        Convert a goal to a problem string.
        """

    @abstractmethod
    def is_action_discrete(self):
        """
        Check if the action space is discrete.
        """

    @abstractmethod
    def is_state_discrete(self):
        """
        Check if the state space is discrete.
        """

    @abstractmethod
    def get_lstm_props(self):
        """
        Get the LSTM properties for the environment.
        """

    @abstractmethod
    def change_done_by_specific_desired(self, obs, desired, old_success_done):
        """
        Change the 'done' flag based on a specific desired goal.
        """

    @abstractmethod
    def is_done(self, done):
        """
        Check if the episode is done.
        """

    @abstractmethod
    def is_success(self, info):
        """
        Check if the episode is successful.
        """

    def create_vec_env(self, kwargs):
        """
        Create a vectorized environment.
        """
        env = gym.make(**kwargs)
        return DummyVecEnv([lambda: env])

    @abstractmethod
    def change_goal_to_specific_desired(self, obs, desired):
        """
        Change the goal to a specific desired goal.
        """


class GCEnvProperty(EnvProperty):
    """
    Base class for goal-conditioned environment properties.
    """

    @abstractmethod
    def use_goal_directed_problem(self):
        """
        Check if the environment uses a goal-directed problem.
        """

    def problem_list_to_str_tuple(self, problems):
        """
        Convert a list of problems to a string tuple.
        """
        return "goal_conditioned"


class MinigridProperty(EnvProperty):
    """
    Environment properties for the Minigrid domain.
    """

    def __init__(self, name):
        super().__init__(name)
        self.domain_name = "minigrid"

    def goal_to_problem_str(self, goal):
        """
        Convert a goal to a problem string.
        """
        return self.name + f"-DynamicGoal-{goal[0]}x{goal[1]}-v0"

    def str_to_goal(self, problem_name):
        """
        Convert a problem name to a goal.
        """
        parts = problem_name.split("-")
        goal_part = [part for part in parts if "x" in part]
        width, height = goal_part[0].split("x")
        return (int(width), int(height))

    def gc_adaptable(self):
        """
        Check if the environment is goal-conditioned adaptable.
        """
        return False

    def problem_list_to_str_tuple(self, problems):
        """
        Convert a list of problems to a string tuple.
        """
        return "_".join([f"[{s.split('-')[-2]}]" for s in problems])

    def is_action_discrete(self):
        """
        Check if the action space is discrete.
        """
        return True

    def is_state_discrete(self):
        """
        Check if the state space is discrete.
        """
        return True

    def get_lstm_props(self):
        """
        Get the LSTM properties for the environment.
        """
        return LSTMProperties(
            batch_size=16, input_size=4, hidden_size=8, num_samples=40000
        )

    def create_sequence_image(self, sequence, img_path, problem_name):
        """
        Create a sequence image for the environment.
        """
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        env_id = (
            problem_name.split("-DynamicGoal-")[0]
            + "-DynamicGoal-"
            + problem_name.split("-DynamicGoal-")[1]
        )
        register(
            id=env_id,
            entry_point="gr_envs.minigrid_scripts.envs:CustomColorEnv",
            kwargs={
                "size": 13 if "Simple" in problem_name else 9,
                "num_crossings": 4 if "Simple" in problem_name else 3,
                "goal_pos": self.str_to_goal(problem_name),
                "obstacle_type": Wall if "Simple" in problem_name else Lava,
                "start_pos": (1, 1) if "Simple" in problem_name else (3, 1),
                "plan": sequence,
            },
        )
        env = gym.make(id=env_id)
        env = RGBImgPartialObsWrapper(env)  # Get pixel observations
        env = ImgObsWrapper(env)  # Get rid of the 'mission' field
        obs, _ = env.reset()  # This now produces an RGB tensor only

        img = env.unwrapped.get_frame()

        ####### save image to file
        image_pil = Image.fromarray(np.uint8(img)).convert("RGB")
        image_pil.save(r"{}.png".format(os.path.join(img_path, "plan_image")))

    def change_done_by_specific_desired(self, obs, desired, old_success_done):
        """
        Change the 'done' flag based on a specific desired goal.
        """
        assert (
            desired is None
        ), "In MinigridProperty, giving a specific 'desired' is not supported."
        return old_success_done

    def is_done(self, done):
        """
        Check if the episode is done.
        """
        assert isinstance(done, np.ndarray)
        return done[0]

    def is_success(self, info):
        """
        Check if the episode is successful.
        """
        raise NotImplementedError("no other option for any of the environments.")

    def change_goal_to_specific_desired(self, obs, desired):
        """
        Change the goal to a specific desired goal.
        """
        assert (
            desired is None
        ), "In MinigridProperty, giving a specific 'desired' is not supported."


class PandaProperty(GCEnvProperty):
    """
    Environment properties for the Panda domain.
    """

    def __init__(self, name):
        """
        Initialize a new instance of the Environment class.

        Args:
            name (str): The name of the environment.

        Attributes:
            domain_name (str): The domain name of the environment.

        """
        super().__init__(name)
        self.domain_name = "panda"

    def str_to_goal(self, problem_name):
        """
        Convert a problem name to a goal.
        """
        try:
            numeric_part = problem_name.split("PandaMyReachDenseX")[1]
            components = [
                component.replace("-v3", "").replace("y", ".").replace("M", "-")
                for component in numeric_part.split("X")
            ]
            floats = []
            for component in components:
                floats.append(float(component))
            return np.array([floats], dtype=np.float32)
        except Exception:
            return "general"

    def goal_to_problem_str(self, goal):
        """
        Convert a goal to a problem string.
        """
        goal_str = "X".join(
            [str(float(g)).replace(".", "y").replace("-", "M") for g in goal[0]]
        )
        return f"PandaMyReachDenseX{goal_str}-v3"

    def gc_adaptable(self):
        """
        Check if the environment is goal-conditioned adaptable.
        """
        return True

    def use_goal_directed_problem(self):
        """
        Check if the environment uses a goal-directed problem.
        """
        return False

    def is_action_discrete(self):
        """
        Check if the action space is discrete.
        """
        return False

    def is_state_discrete(self):
        """
        Check if the state space is discrete.
        """
        return False

    def get_lstm_props(self):
        """
        Get the LSTM properties for the environment.
        """
        return LSTMProperties(
            batch_size=32, input_size=9, hidden_size=8, num_samples=20000
        )

    def sample_goal():
        """
        Sample a random goal.
        """
        goal_range_low = np.array([-0.40, -0.40, 0.10])
        goal_range_high = np.array([0.2, 0.2, 0.10])
        return np.random.uniform(goal_range_low, goal_range_high)

    def change_done_by_specific_desired(self, obs, desired, old_success_done):
        """
        Change the 'done' flag based on a specific desired goal.
        """
        if desired is None:
            return old_success_done
        assert isinstance(
            desired, np.ndarray
        ), f"Unsupported type for desired: {type(desired)}"
        if desired.size > 0 and not np.isnan(desired).all():
            assert (
                obs["achieved_goal"].shape == desired.shape
            ), f"Shape mismatch: {obs['achieved_goal'].shape} vs {desired.shape}"
            d = np.linalg.norm(obs["achieved_goal"] - desired, axis=-1)
            return (d < 0.04)[0]
        else:
            return old_success_done

    def is_done(self, done):
        """
        Check if the episode is done.
        """
        assert isinstance(done, np.ndarray)
        return done[0]

    def is_success(self, info):
        """
        Check if the episode is successful.
        """
        assert "is_success" in info[0].keys()
        return info[0]["is_success"]

    def change_goal_to_specific_desired(self, obs, desired):
        """
        Change the goal to a specific desired goal.
        """
        if desired is not None:
            obs["desired_goal"] = desired


class ParkingProperty(GCEnvProperty):
    """
    Environment properties for the Parking domain.
    """

    def __init__(self, name):
        """
        Initialize a new environment object.

        Args:
            name (str): The name of the environment.

        Attributes:
            domain_name (str): The domain name of the environment.

        """
        super().__init__(name)
        self.domain_name = "parking"

    def goal_to_problem_str(self, goal):
        """
        Convert a goal to a problem string.
        """
        return self.name.split("-v0")[0] + f"-GI-{goal}-v0"

    def gc_adaptable(self):
        """
        Check if the environment is goal-conditioned adaptable.
        """
        return True

    def is_action_discrete(self):
        """
        Check if the action space is discrete.
        """
        return False

    def is_state_discrete(self):
        """
        Check if the state space is discrete.
        """
        return False

    def use_goal_directed_problem(self):
        """
        Check if the environment uses a goal-directed problem.
        """
        return True

    def get_lstm_props(self):
        """
        Get the LSTM properties for the environment.
        """
        return LSTMProperties(
            batch_size=32, input_size=8, hidden_size=8, num_samples=20000
        )

    def change_done_by_specific_desired(self, obs, desired, old_success_done):
        """
        Change the 'done' flag based on a specific desired goal.
        """
        assert (
            desired is None
        ), "In ParkingProperty, giving a specific 'desired' is not supported."
        return old_success_done

    def is_done(self, done):
        """
        Check if the episode is done.
        """
        assert isinstance(done, np.ndarray)
        return done[0]

    def is_success(self, info):
        """
        Check if the episode is successful.
        """
        assert "is_success" in info[0].keys()
        return info[0]["is_success"]

    def change_goal_to_specific_desired(self, obs, desired):
        """
        Change the goal to a specific desired goal.
        """
        assert (
            desired is None
        ), "In ParkingProperty, giving a specific 'desired' is not supported."


class PointMazeProperty(EnvProperty):
    """Environment properties for the Point Maze domain."""

    def __init__(self, name):
        """
        Initializes a new instance of the Environment class.

        Args:
            name (str): The name of the environment.

        Attributes:
            domain_name (str): The domain name of the environment.
        """
        super().__init__(name)
        self.domain_name = "point_maze"

    def str_to_goal(self):
        """Convert a problem name to a goal."""
        parts = self.name.split("-")
        # Find the part containing the goal size (usually after "DynamicGoal")
        sizes_parts = [part for part in parts if "x" in part]
        goal_part = sizes_parts[1]
        # Extract width and height from the goal part
        width, height = goal_part.split("x")
        return (int(width), int(height))

    def gc_adaptable(self):
        """Check if the environment is goal-conditioned adaptable."""
        return False

    def problem_list_to_str_tuple(self, problems):
        """Convert a list of problems to a string tuple."""
        return "_".join([f"[{s.split('-')[-1]}]" for s in problems])

    def is_action_discrete(self):
        """Check if the action space is discrete."""
        return False

    def is_state_discrete(self):
        """Check if the state space is discrete."""
        return False

    def get_lstm_props(self):
        """
        Get the LSTM properties for the environment.
        """
        return LSTMProperties(
            batch_size=32, input_size=6, hidden_size=8, num_samples=20000
        )

    def goal_to_problem_str(self, goal):
        """
        Convert a goal to a problem string.
        """
        return self.name + f"-Goal-{goal[0]}x{goal[1]}"

    def change_done_by_specific_desired(self, obs, desired, old_success_done):
        """
        Change the 'done' flag based on a specific desired goal.
        """
        assert (
            desired is None
        ), "In PointMazeProperty, giving a specific 'desired' is not supported."
        return old_success_done

    def is_done(self, done):
        """
        Check if the episode is done.
        """
        assert isinstance(done, np.ndarray)
        return done[0]

    def is_success(self, info):
        """
        Check if the episode is successful.
        """
        assert "success" in info[0].keys()
        return info[0]["success"]

    def change_goal_to_specific_desired(self, obs, desired):
        """
        Change the goal to a specific desired goal.
        """
        assert (
            desired is None
        ), "In ParkingProperty, giving a specific 'desired' is not supported."
