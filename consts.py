from metrics.metrics import mean_wasserstein_distance
from environment.utils import maze_str_to_goal, parking_str_to_goal, minigrid_str_to_goal
from ml.tabular.tabular_q_learner import TabularQLearner
from recognizer.graml.graml_recognizer import MCTS_BASED, AGENT_BASED, GC_AGENT_BASED
from ml.neural.SB3model import NeuralAgent
from stable_baselines3 import SAC, TD3

MINIGRID_PROBLEMS = {
    'MiniGrid-Empty-9x9-2-PROBLEMS':
        (
            'MiniGrid-Empty-9x9-v0',
            [
                "MiniGrid-DynamicGoalEmpty-9x9-7x7-v0",
                "MiniGrid-DynamicGoalEmpty-9x9-7x1-v0"
            ]
        )
    ,
    'MiniGrid-Empty-9x9-3-PROBLEMS':
        (
            'MiniGrid-Empty-9x9-v0',
            [
                "MiniGrid-DynamicGoalEmpty-9x9-7x7-v0",
                "MiniGrid-DynamicGoalEmpty-9x9-7x1-v0",
                "MiniGrid-DynamicGoalEmpty-9x9-4x7-v0"
            ]
        )
    ,
    'MiniGrid-Lava-7x7-3-PROBLEMS':
        (
            'MiniGrid-Lava-7x7-v0',
            [
                "MiniGrid-LavaCrossingS9N1-DynamicGoal-7x7-v0", "MiniGrid-LavaCrossingS9N1-DynamicGoal-1x3-v0",
                "MiniGrid-LavaCrossingS9N1-DynamicGoal-7x1-v0"
            ]
        )
    ,
    'MiniGrid-Empty-8x8-3-PROBLEMS':
        (
            'MiniGrid-Empty-8x8-v0',
            [
                # "MiniGrid-DynamicGoalEmpty-8x8-6x6-v0",
                # "MiniGrid-DynamicGoalEmpty-8x8-6x1-v0",
                # "MiniGrid-DynamicGoalEmpty-8x8-1x6-v0",
                "MiniGrid-DynamicGoalEmpty-8x8-3x6-v0",
                "MiniGrid-DynamicGoalEmpty-8x8-4x4-v0",
                "MiniGrid-DynamicGoalEmpty-8x8-6x3-v0",
                "MiniGrid-DynamicGoalEmpty-8x8-1x3-v0"
            ]
        )
    ,
    'MiniGrid-Simple-9x9-3-PROBLEMS':
        (
            'MiniGrid-Simple-9x9-v0',
            [
                # "MiniGrid-SimpleCrossingS9N2-DynamicGoal-7x1-v0",
                # "MiniGrid-SimpleCrossingS9N2-DynamicGoal-1x7-v0",
                # "MiniGrid-SimpleCrossingS9N2-DynamicGoal-7x7-v0",
                "MiniGrid-SimpleCrossingS9N2-DynamicGoal-6x5-v0",
                "MiniGrid-SimpleCrossingS9N2-DynamicGoal-2x3-v0",
                "MiniGrid-SimpleCrossingS9N2-DynamicGoal-3x7-v0",
                "MiniGrid-SimpleCrossingS9N2-DynamicGoal-4x1-v0"
            ]
        )
    ,
    
    'MiniGrid-Walls-13x13-10-PROBLEMS':
        (
            'MiniGrid-Walls-13x13-v0',
            [
                "MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",
                "MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",
                "MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",
                # "MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x3-v0",
                "MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",
                # "MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x5-v0",
                # "MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x8-v0",
                "MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",
                # "MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x1-v0",
                # "MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x7-v0",
                # "MiniGrid-SimpleCrossingS13N4-DynamicGoal-5x9-v0"
            ]
        )
    ,
}

MAZE_PROBLEMS = {
    'PointMaze-FourRoomsEnv-11x11-3-PROBLEMS':
        (
            'PointMaze-FourRoomsEnvDense-11x11',
            [
                "PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",
                "PointMaze-FourRoomsEnv-11x11-Goal-9x9", # this one doesn't work with dense rewards because of encountering local minima
                "PointMaze-FourRoomsEnvDense-11x11-Goal-1x9"
            ]
        )
    ,
    'PointMaze-ObstaclesEnv-11x11-2-PROBLEMS':
        (
            'PointMaze-ObstaclesEnv-11x11',
            [
                "PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",
                #"PointMaze-ObstaclesEnv-11x11-Goal-9x9", # this one doesn't work with dense rewards because of encountering local minima
                "PointMaze-ObstaclesEnvDense-11x11-Goal-1x5"
            ]
        )
    ,
    'PointMaze-ObstaclesEnv-11x11-3-PROBLEMS':
        (
            'PointMaze-ObstaclesEnv-11x11',
            [
                "PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-9x9", # this one doesn't work with dense rewards because of encountering local minima
                "PointMaze-ObstaclesEnvDense-11x11-Goal-1x5"
            ],
        )
    ,
    'PointMaze-ObstaclesEnv1-11x11-3-PROBLEMS':
        (
            'PointMaze-ObstaclesEnv1-11x11',
            [
                "PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-9x9", # this one doesn't work with dense rewards because of encountering local minima
                "PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-7x4"
            ]
        )
    ,
    'PointMaze-ObstaclesEnv2-11x11-3-PROBLEMS':
        (
            'PointMaze-ObstaclesEnv2-11x11',
            [
                "PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-9x9", # this one doesn't work with dense rewards because of encountering local minima
                "PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-7x4"
            ]
        )
    ,
    'PointMaze-ObstaclesEnv3-11x11-3-PROBLEMS':
        (
            'PointMaze-ObstaclesEnv3-11x11',
            [
                "PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-9x9", # this one doesn't work with dense rewards because of encountering local minima
                "PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-7x4"
            ]
        )
    ,
    'PointMaze-ObstaclesEnv4-11x11-3-PROBLEMS':
        (
            'PointMaze-ObstaclesEnv4-11x11',
            [
                "PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-9x9", # this one doesn't work with dense rewards because of encountering local minima
                "PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-6x6"
            ]
        )
    ,
    'PointMaze-ObstaclesEnv5-11x11-3-PROBLEMS':
        (
            'PointMaze-ObstaclesEnv5-11x11',
            [
                "PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-9x9", # this one doesn't work with dense rewards because of encountering local minima
                "PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",
                "PointMaze-ObstaclesEnvDense-11x11-Goal-7x6"
            ]
        )
    ,
}

PARKING_PROBLEMS = {
    'ParkingEnvContinuous-Hard-4-Problems':
        (
            'ParkingEnv',  # OBS
            [
                "Parking-S-14-PC--GI-4-v0",
                "Parking-S-14-PC--GI-8-v0",
                "Parking-S-14-PC--GI-13-v0",
                "Parking-S-14-PC--GI-17-v0",
                "Parking-S-14-PC--GI-21-v0",
            ],
        )
    ,
    'ParkingEnvUniversal-Hard-4-Problems':
        (
            'ParkingEnv',  # must be different
            [
                "parking-v0",
            ],
        )
    ,
}

PROBLEMS = {
    "point_maze":
        {
            "additional_combined_recognizer_kwargs": {
                "learner_type": NeuralAgent,
                "specified_rl_algorithm": SAC,
                "specified_rl_algorithm_inference": TD3
            },
            "tasks": {
                "obstacles":
                    {
                        "L1": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            #("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L2": {
                            "base_goals_problems": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L3": {
                            "base_goals_problems": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L4": {
                            "base_goals_problems": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L5": {
                            "base_goals_problems": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x7",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        }
                    },
                "four_rooms":
                    {
                        "L1": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L2": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-ObstaclesEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L3": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-ObstaclesEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x8",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L4": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-ObstaclesEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x8",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x3",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L5": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-ObstaclesEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x8",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x3",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x7",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        }
                    },
            }
        },
        "minigrid":
        {
            "additional_combined_recognizer_kwargs": {
                "learner_type": TabularQLearner
            },
            "tasks": {
                "obstacles":
                    {
                        "L1": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            #("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L2": {
                            "base_goals_problems": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L3": {
                            "base_goals_problems": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L4": {
                            "base_goals_problems": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L5": {
                            "base_goals_problems": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x7",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        }
                    },
                "four_rooms":
                    {
                        "L1": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L2": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-ObstaclesEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L3": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-ObstaclesEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x8",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L4": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-ObstaclesEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x8",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x3",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L5": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-ObstaclesEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x8",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x3",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x7",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-8x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 6,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "is_inference_same_length_sequences": {"inference_same_length": True, "inference_diff_length": False},
                                "is_learn_same_length_sequences": {"learn_same_length": True, "learn_diff_length": False},
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        }
                    },
            }
}

# agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["kettle", "microwave"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["bottom burner", "top burner"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["light switch", "microwave"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["slide cabinet", "hinge cabinet"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["bottom burner", "microwave"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["kettle", "light switch"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()
    # agent = NeuralAgent(env_name="FrankaKitchen-v1", problem_name="FrankaKitchen-v1", tasks_to_complete = ["bottom burner", "hinge cabinet"], algorithm=TD3, num_timesteps=400000, complex_obs_space=True)
    # agent.learn()

KITCHEN_PROBLEMS = {
    'FrankaKitchen-4-Problems':
        (
            'FrankaKitchen-v1',  # OBS
            [
                "kettle",
                "slide cabinet",
                "bottom burner",
                "light switch",
            ],
        )
    ,
    'ParkingEnvUniversal-Hard-4-Problems':
        (
            'ParkingEnv',  # must be different
            [
                "parking-v0",
            ],
        )
    ,
}
# python graml_main.py graml fragmented_partial_obs inference_same_length learn_diff_length collect_statistics MAZE:OBSTACLES