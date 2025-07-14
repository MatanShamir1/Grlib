"""environment.py"""

import os
import sys
from abc import abstractmethod
from collections import namedtuple
from contextlib import contextmanager

import gymnasium as gym
import numpy as np
from gr_envs.wrappers.goal_wrapper import GoalRecognitionWrapper
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


@contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr (including C/C++ prints).
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


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
    def goal_to_str(self, goal):
        """
        Convert a goal to a string representation.
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
        Create a vectorized environment, suppressing prints from gym/pybullet/panda-gym.
        """
        with suppress_output():
            env = gym.make(**kwargs)
        return DummyVecEnv([lambda: env])

    @abstractmethod
    def change_goal_to_specific_desired(self, obs, desired):
        """
        Change the goal to a specific desired goal.
        """

    def is_goal_in_subspace(self, goal):
        """
        Check if a goal is within the specified goal subspace.

        Args:
            goal: The goal to check
            goal_subspace: The goal subspace to check against

        Returns:
            bool: True if the goal is within the subspace, False otherwise
        """
        env = gym.make(id=self.name)
        while env is not None and hasattr(env, "env"):
            if isinstance(env, GoalRecognitionWrapper) and hasattr(
                env, "is_goal_in_subspace"
            ):
                # If the environment has a goal recognition wrapper, use its method
                return env.is_goal_in_subspace(goal)
            # Traverse through wrappers to find the base environment
            env = env.env

        return True


class GCEnvProperty(EnvProperty):
    """
    Base class for goal-conditioned environment properties.
    """

    @abstractmethod
    def use_goal_directed_problem(self):
        """
        Check if the environment uses a goal-directed problem.
        """


class MinigridProperty(EnvProperty):
    """
    Environment properties for the Minigrid domain.
    """

    def __init__(self, name):
        super().__init__(name)
        self.domain_name = "minigrid"

    def goal_to_str(self, goal):
        """
        Convert a goal to a string representation.
        """
        return f"{goal[0]}x{goal[1]}"

    def goal_to_problem_str(self, goal):
        """
        Convert a goal to a problem string.
        """
        return self.name + f"-DynamicGoal-{self.goal_to_str(goal)}-v0"

    def str_to_goal(self, problem_name=None):
        """
        Convert a problem name to a goal.
        """
        if problem_name is None:
            problem_name = self.name

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
        if not os.path.exists(img_path):
            os.makedirs(img_path)
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

    def str_to_goal(self, problem_name=None):
        """
        Convert a problem name to a goal.
        """
        if problem_name is None:
            return "general"
        try:
            numeric_part = problem_name.split("PandaMyReachDenseX")[1]
            components = [
                component.replace("-v3", "").replace("y", ".").replace("M", "-")
                for component in numeric_part.split("X")
            ]
            floats = [float(component) for component in components]
            return np.array([floats])
        except Exception:
            return "general"

    def goal_to_str(self, goal):
        """
        Convert a goal to a string representation.
        """
        return "X".join(
            [str(float(g)).replace(".", "y").replace("-", "M") for g in goal[0]]
        )

    def goal_to_problem_str(self, goal):
        """
        Convert a goal to a problem string.
        """
        goal_str = self.goal_to_str(goal)
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

    def problem_list_to_str_tuple(self, problems):
        """
        Convert a list of problems to a string tuple.
        """
        if not isinstance(problems, list) or len(problems) == 1:
            return "goal_conditioned"

        def decode_goal_str(goal_str):
            # Split by X, decode each component
            components = [
                float(s.replace("y", ".").replace("M", "-"))
                for s in goal_str.split("X")
            ]
            return components

        goal_strs = []
        for s in problems:
            # Extract the encoded goal part between 'PandaMyReachDenseX' and '-v3'
            encoded = s.split("PandaMyReachDenseX")[1].split("-v3")[0]
            decoded = decode_goal_str(encoded)
            goal_strs.append(f"[{','.join(str(x) for x in decoded)}]")

        return "_".join(goal_strs)


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

    def str_to_goal(self, problem_name=None):
        """
        Convert a problem name to a goal.
        """
        if not problem_name:
            problem_name = self.name
        # Extract the goal from the part
        return int(problem_name.split("GI-")[1].split("-v0")[0])

    def goal_to_str(self, goal):
        """
        Convert a goal to a string representation.
        """
        if isinstance(goal, int):
            return str(goal)
        elif isinstance(goal, str):
            return goal
        else:
            raise ValueError(
                f"Unsupported goal type: {type(goal)}. Expected int or str."
            )

    def goal_to_problem_str(self, goal):
        """
        Convert a goal to a problem string.
        """
        if "-GI-" in self.name:
            return self.name.split("-GI-")[0] + f"-GI-{goal}-v0"
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

    def problem_list_to_str_tuple(self, problems):
        """
        Convert a list of problems to a string tuple.
        """
        if (not isinstance(problems, list)) or len(problems) == 1:
            return "goal_conditioned"
        return "_".join([f"[{s.split('-GI-')[-1].split('-v0')[0]}]" for s in problems])


class PointMazeProperty(GCEnvProperty):
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

    def str_to_goal(self, problem_name=None):
        """Convert a problem name to a goal."""
        if not problem_name:
            problem_name = self.name
        parts = problem_name.split("-")
        # Find the part containing the goal size (usually after "DynamicGoal")
        sizes_parts = [part for part in parts if "x" in part]
        goal_part = sizes_parts[1]
        # Extract width and height from the goal part
        width, height = goal_part.split("x")
        return (int(width), int(height))

    def goal_to_str(self, goal):
        """
        Convert a goal to a string representation.
        """
        return f"{goal[0]}x{goal[1]}"

    def gc_adaptable(self):
        """Check if the environment is goal-conditioned adaptable."""
        return True

    def problem_list_to_str_tuple(self, problems):
        """Convert a list of problems to a string tuple."""
        if not isinstance(problems, list) or len(problems) == 1:
            return "goal_conditioned"
        else:
            return "_".join([f"[{s.split('-')[-1]}]" for s in problems])

    def use_goal_directed_problem(self):
        """
        Check if the environment uses a goal-directed problem.
        """
        return True

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
            batch_size=32, input_size=6, hidden_size=8, num_samples=10000
        )

    def goal_to_problem_str(self, goal):
        """
        Convert a goal to a problem string.
        """
        possible_suffixes = ["-Goals-", "-Goal-", "-MultiGoals-", "-GoalConditioned-"]
        for suffix in possible_suffixes:
            if suffix in self.name:
                return self.name.split(suffix)[0] + f"-Goal-{self.goal_to_str(goal)}"

        return self.name + f"-Goal-{self.goal_to_str(goal)}"

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
        ), "In PointMazeProperty, giving a specific 'desired' is not supported."

    def change_done_by_specific_desired(self, obs, desired, old_success_done):
        """
        Change the 'done' flag based on a specific desired goal.
        """
        assert (
            desired is None
        ), "In ParkingProperty, giving a specific 'desired' is not supported."
        return old_success_done
