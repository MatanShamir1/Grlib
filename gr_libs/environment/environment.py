from abc import abstractmethod
from collections import namedtuple
import os

import gymnasium
from PIL import Image
import numpy as np
from gymnasium.envs.registration import register
from minigrid.core.world_object import Wall, Lava
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

MINIGRID, PANDA, PARKING, POINT_MAZE = "minigrid", "panda", "parking", "point_maze"

QLEARNING = "QLEARNING"

SUPPORTED_DOMAINS = [MINIGRID, PANDA, PARKING, POINT_MAZE]

LSTMProperties = namedtuple('LSTMProperties', ['input_size', 'hidden_size', 'batch_size', 'num_samples'])



class EnvProperty:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"{self.name}"

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)
    
    @abstractmethod
    def str_to_goal(self):
        pass

    @abstractmethod
    def gc_adaptable(self):
        pass

    @abstractmethod
    def problem_list_to_str_tuple(self, problems):
        pass

    @abstractmethod
    def goal_to_problem_str(self, goal):
        pass

    @abstractmethod
    def is_action_discrete(self):
        pass

    @abstractmethod
    def is_state_discrete(self):
        pass

    @abstractmethod
    def get_lstm_props(self):
        pass

class GCEnvProperty(EnvProperty):
    @abstractmethod
    def use_goal_directed_problem(self):
        pass

    def problem_list_to_str_tuple(self, problems):
        return "goal_conditioned"

class MinigridProperty(EnvProperty):
    def __init__(self, name):
        super().__init__(name)
        self.domain_name = "minigrid"

    def goal_to_problem_str(self, goal):
        return self.name + f"-DynamicGoal-{goal[0]}x{goal[1]}-v0"

    def str_to_goal(self, problem_name):
        parts = problem_name.split("-")
        goal_part = [part for part in parts if "x" in part]
        width, height = goal_part[0].split("x")
        return (int(width), int(height))

    def gc_adaptable(self):
        return False
    
    def problem_list_to_str_tuple(self, problems):
        return "_".join([f"[{s.split('-')[-2]}]" for s in problems])
    
    def is_action_discrete(self):
        return True

    def is_state_discrete(self):
        return True

    def get_lstm_props(self):
        return LSTMProperties(batch_size=16, input_size=4, hidden_size=8, num_samples=40000)
    
    def create_sequence_image(self, sequence, img_path, problem_name):
        if not os.path.exists(os.path.dirname(img_path)): os.makedirs(os.path.dirname(img_path))
        env_id = problem_name.split("-DynamicGoal-")[0] + "-DynamicGoal-" + problem_name.split("-DynamicGoal-")[1]
        result = register(
            id=env_id,
            entry_point="gr_envs.minigrid_scripts.envs:CustomColorEnv",
            kwargs={"size": 13 if 'Simple' in problem_name else 9,
                    "num_crossings": 4 if 'Simple' in problem_name else 3,
                    "goal_pos": self.str_to_goal(problem_name),
                    "obstacle_type": Wall if 'Simple' in problem_name else Lava,
                    "start_pos": (1, 1) if 'Simple' in problem_name else (3, 1),
                    "plan": sequence},
        )
        #print(result)
        env = gymnasium.make(id=env_id)
        env = RGBImgPartialObsWrapper(env) # Get pixel observations
        env = ImgObsWrapper(env) # Get rid of the 'mission' field
        obs, _ = env.reset() # This now produces an RGB tensor only

        img = env.unwrapped.get_frame()

        ####### save image to file
        image_pil = Image.fromarray(np.uint8(img)).convert('RGB')
        image_pil.save(r"{}.png".format(img_path))

    
class PandaProperty(GCEnvProperty):
    def __init__(self, name):
        super().__init__(name)
        self.domain_name = "panda"

    def str_to_goal(self, problem_name):
        try:
            numeric_part = problem_name.split('PandaMyReachDenseX')[1]
            components = [component.replace('-v3', '').replace('y', '.').replace('M', '-') for component in numeric_part.split('X')]
            floats = []
            for component in components:
                floats.append(float(component))
            return np.array([floats], dtype=np.float32)
        except Exception as e:
            return "general"
        
    def goal_to_problem_str(self, goal):
        goal_str = 'X'.join([str(float(g)).replace(".", "y").replace("-","M") for g in goal[0]])
        return f"PandaMyReachDenseX{goal_str}-v3"

    def gc_adaptable(self):
        return True
    
    def use_goal_directed_problem(self):
        return False
    
    def is_action_discrete(self):
        return False

    def is_state_discrete(self):
        return False

    def get_lstm_props(self):
        return LSTMProperties(batch_size=32, input_size=9, hidden_size=8, num_samples=20000)
    
    def sample_goal():
        goal_range_low = np.array([-0.40, -0.40, 0.10])
        goal_range_high = np.array([0.2, 0.2, 0.10])
        return np.random.uniform(goal_range_low, goal_range_high)

        
class ParkingProperty(GCEnvProperty):

    def __init__(self, name):
        super().__init__(name)
        self.domain_name = "parking"

    def goal_to_problem_str(self, goal):
        return self.name.split("-v0")[0] + f"-GI-{goal}-v0"

    def gc_adaptable(self):
        return True
    
    def is_action_discrete(self):
        return False

    def is_state_discrete(self):
        return False
    
    def use_goal_directed_problem(self):
        return True
    
    def get_lstm_props(self):
        return LSTMProperties(batch_size=32, input_size=8, hidden_size=8, num_samples=20000)


class PointMazeProperty(EnvProperty):
    def __init__(self, name):
        super().__init__(name)
        self.domain_name = "point_maze"

    def str_to_goal(self):
        parts = self.name.split("-")
        # Find the part containing the goal size (usually after "DynamicGoal")
        sizes_parts = [part for part in parts if "x" in part]
        goal_part = sizes_parts[1]
        # Extract width and height from the goal part
        width, height = goal_part.split("x")
        return (int(width), int(height))
    
    def gc_adaptable(self):
        return False

    def problem_list_to_str_tuple(self, problems):
        return "_".join([f"[{s.split('-')[-1]}]" for s in problems])

    def is_action_discrete(self):
        return False

    def is_state_discrete(self):
        return False
    
    def get_lstm_props(self):
        return LSTMProperties(batch_size=32, input_size=6, hidden_size=8, num_samples=20000)

    def goal_to_problem_str(self, goal):
        return self.name + f"-Goal-{goal[0]}x{goal[1]}"
