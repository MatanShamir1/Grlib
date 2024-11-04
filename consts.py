import numpy as np
from grlib.metrics.metrics import kl_divergence_norm_softmax, mean_wasserstein_distance
from grlib.environment.utils import maze_str_to_goal, parking_str_to_goal, minigrid_str_to_goal, panda_str_to_goal
from grlib.ml.tabular.tabular_q_learner import TabularQLearner
from grlib.recognizer.graml.graml_recognizer import MCTS_BASED, AGENT_BASED, GC_AGENT_BASED
from gr_libs.panda_gym_scripts.envs.tasks.my_reach import MyReach
from grlib.ml.neural.SB3model import NeuralAgent
from stable_baselines3 import PPO, SAC, TD3

def sample_goal():
    goal_range_low = np.array([-0.40, -0.40, 0.10])
    goal_range_high = np.array([0.2, 0.2, 0.10])
    return np.random.uniform(goal_range_low, goal_range_high)

def sample_goal_panda_problem():
    goal = sample_goal()
    goal_str = 'PandaMyReachDense' + 'X'.join([str(float(g)).replace(".", "y").replace("-","M") for g in goal]) + "-v3"
    return goal_str

PROBLEMS = {
        "point_maze":
        {
            "additional_combined_recognizer_kwargs": {
                "learner_type": NeuralAgent,
                "specified_rl_algorithm_learning": SAC,
                "specified_rl_algorithm_inference": TD3
            },
            "tasks": {
                "obstacles": # relevants for increasing number of dynamic goals are L111-L555, increasing base goals L1-L5, static goal recognition L11-L55
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
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                                "specified_rl_algorithm_adaptation": SAC,
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
                                "specified_rl_algorithm_adaptation": SAC,
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
                            "base_goals_problems_train_configs": [
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
                                "specified_rl_algorithm_adaptation": SAC,
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
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L11": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L22": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L33": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 300000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L44": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 300000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L55": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x7",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 300000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x7",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L111": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x7",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            #("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            # ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            #("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L222": {
                            "base_goals_problems_train_configs": [
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
                                                            # ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            #("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            #("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L333": {
                            "base_goals_problems_train_configs": [
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
                                                            #("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            #("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L444": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-5x1",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-9x9",(None, 200000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-1x5",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x4",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x6",(None, 200000)),
                                                    ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x7",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            #("PointMaze-ObstaclesEnvDense-11x11-Goal-5x5",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-3x6",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-6x3",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-4x7",(None, 200000)),
                                                            ("PointMaze-ObstaclesEnvDense-11x11-Goal-8x8",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L555": {
                           "base_goals_problems_train_configs": [
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
                                "specified_rl_algorithm_adaptation": SAC,
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
                "four_rooms": # relevant for increasing number of dynamic goals are L111-L555, increasing base goals L1-L5, static goal recognition L11-L55
                    {
                        "L111": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x3",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L222": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x3",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L333": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x3",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L444": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                            # ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x3",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L555": {
                           "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4",(None, 200000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-4x3",(None, 200000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                        "L11": {
                            "base_goals_problems_train_configs": [
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                    ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1",(None, 300000)),
                                                            ("PointMaze-FourRoomsEnv-11x11-Goal-9x9",(None, 400000)), # this one doesn't work with dense rewards because of encountering local minima
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9",(None, 300000)),
                                                            ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3",(None, 300000))
                                                           ],
                            "task_str_to_goal": maze_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC,
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
                            }
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
                "obstacles": # relevants for increasing number of dynamic goals are L111-L555, increasing base goals L1-L5, static goal recognition L11-L55
                    {
                        "L1": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x3-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x5-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x8-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x7-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-5x9-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": MCTS_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L2": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x3-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x5-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x8-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x7-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-5x9-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": MCTS_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L3": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x3-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x5-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x8-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x7-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-5x9-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": MCTS_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L4": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x3-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x5-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x8-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x7-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-5x9-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": MCTS_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L5": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0",(None, None))
                                                    
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x3-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x5-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x8-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x7-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-5x9-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": MCTS_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L11": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L22": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L33": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                           ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L44": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L55": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L111": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L222": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L333": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                           ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L444": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L555": { # python experiments.py --recognizer graml --domain minigrid --task L555 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats --inference_same_seq_len
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None)),
                                                    ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0",(None, None)),
                                                            ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                    },
                "lava_crossing": # relevant for increasing number of dynamic goals are L111-L555, increasing base goals L1-L5, static goal recognition L11-L55
                    {
                        "L111": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x1-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L222": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x1-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L333": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x1-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L444": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x1-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L555": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-7x1-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L11": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L22": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L33": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L44": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                            #("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                        "L55": {
                            "base_goals_problems_train_configs": [
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                    ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x3-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-6x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-2x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-5x2-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-4x5-v0",(None, None)),
                                                            ("MiniGrid-LavaCrossingS9N2-DynamicGoal-1x1-v0",(None, None))
                                                           ],
                            "task_str_to_goal": minigrid_str_to_goal,
                            "additional_graml_kwargs": {
                                "input_size": 4,
                                "hidden_size": 8,
                                "batch_size": 16,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems]),
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": kl_divergence_norm_softmax
                            },
                        },
                    }
                }
        },
        "parking":
        {
            "additional_combined_recognizer_kwargs": {
                "learner_type": NeuralAgent,
                "specified_rl_algorithm_learning": PPO,
                "specified_rl_algorithm_inference": TD3,
                "use_goal_directed_problem": True
            },
            "tasks": {
                "gc_agent":
                    {
                        "L111": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("parking-v0",(None, 400000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            #("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": PPO, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [f"Parking-S-14-PC--GI-{i}-v0" for i in range(1,21)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L222": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("parking-v0",(None, 400000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            #("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": PPO, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [f"Parking-S-14-PC--GI-{i}-v0" for i in range(1,21)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L333": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("parking-v0",(None, 400000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            #("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": PPO, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [f"Parking-S-14-PC--GI-{i}-v0" for i in range(1,21)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L444": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("parking-v0",(None, 400000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            #("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            #("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": PPO, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [f"Parking-S-14-PC--GI-{i}-v0" for i in range(1,21)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L555": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("parking-v0",(None, 400000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": PPO, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [f"Parking-S-14-PC--GI-{i}-v0" for i in range(1,21)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        }
                    },
                "gd_agent":
                    {
                        "L1": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L2": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L3": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L4": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L5": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-20-v0",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-1-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-4-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-8-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-11-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-14-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-18-v0",(None, 400000)),
                                                            ("Parking-S-14-PC--GI-21-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L11": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L22": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L33": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L44": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L55": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-20-v0",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-20-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L111": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-20-v0",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L222": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-20-v0",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L333": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-20-v0",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L444": {
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-20-v0",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            #("Parking-S-14-PC--GI-20-v0",(None, 200000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                        "L555": { # python experiments.py --recognizer graml --domain parking --task L555 --partial_obs_type fragmented --parking_env gd_agent --collect_stats --inference_same_seq_len
                            "base_goals_problems_train_configs": [
                                                    ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                    ("Parking-S-14-PC--GI-20-v0",(None, 300000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("Parking-S-14-PC--GI-3-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-7-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-10-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-13-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-15-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-17-v0",(None, 200000)),
                                                            ("Parking-S-14-PC--GI-20-v0",(None, 300000))
                                                           ],
                            "task_str_to_goal": parking_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gd_agent, mustn't be the same as specified_rl_algorithm_learning
                                "input_size": 8,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": AGENT_BASED,
                                "gc_sequence_generation": False,
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{s.split('-')[-2]}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance
                            },
                        },
                    },
            }
        },
        "panda":
        {
            "additional_combined_recognizer_kwargs": {
                "learner_type": NeuralAgent,
                "specified_rl_algorithm_learning": SAC,
                "specified_rl_algorithm_inference": PPO,
                "use_goal_directed_problem": False
            },
            "tasks": {
                "gc_agent":
                    {
                        "L111": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDense-v3",(None, 800000))
                                                    ],
                            "dynamic_goals_train_configs": [ # goals = [(-0.5, -0.5, 0.1), (-0.3, -0.3, 0.1), (-0.1, -0.1, 0.1), (-0.5, 0.2, 0.1), (-0.3, 0.2, 0.1), (-0.1, 0.1, 0.1), (0.2, -0.2, 0.1), (0.2, -0.3, 0.1), (0.1, -0.1, 0.1), (0.2, 0.2, 0.1), (0.0, 0.0, 0.1), (0.1, 0.1, 0.1)]
                                                            #("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 200000)),
                                                            # 3 similar:
                                                            #("PandaMyReachDenseXM0y5X0y2X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y3X0y2X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y1X0y1X0y1-v3",(None, 200000)),
                                                            # 3 similar:
                                                            #("PandaMyReachDenseX0y2XM0y2X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseX0y1XM0y1X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 200000)),
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L222": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDense-v3",(None, 800000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 400000)),
                                                            #("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 400000)),
                                                            #("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 400000)),
                                                            # 3 similar:
                                                            ("PandaMyReachDenseXM0y5X0y2X0y1-v3",(None, 400000)),
                                                            #("PandaMyReachDenseXM0y3X0y2X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y1X0y1X0y1-v3",(None, 400000)),
                                                            # 3 similar
                                                            #("PandaMyReachDenseX0y2XM0y2X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseX0y1XM0y1X0y1-v3",(None, 400000)),
                                                            #("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 400000))
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L333": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDense-v3",(None, 800000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 400000)),
                                                            #("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 400000)),
                                                            # 3 similar:
                                                            ("PandaMyReachDenseXM0y5X0y2X0y1-v3",(None, 400000)),
                                                            #("PandaMyReachDenseXM0y3X0y2X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y1X0y1X0y1-v3",(None, 400000)),
                                                            # 3 similar
                                                            ("PandaMyReachDenseX0y2XM0y2X0y1-v3",(None, 400000)),
                                                            #("PandaMyReachDenseX0y1XM0y1X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 400000))
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L444": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDense-v3",(None, 800000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            #("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 400000)),
                                                            # 3 similar:
                                                            #("PandaMyReachDenseXM0y5X0y2X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y3X0y2X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y1X0y1X0y1-v3",(None, 400000)),
                                                            # 3 similar4
                                                            #("PandaMyReachDenseX0y2XM0y2X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseX0y1XM0y1X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 400000))
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L555": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDense-v3",(None, 800000))
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 400000)),
                                                            # 3 similar:
                                                            ("PandaMyReachDenseXM0y5X0y2X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y3X0y2X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseXM0y1X0y1X0y1-v3",(None, 400000)),
                                                            # 3 similar4
                                                            ("PandaMyReachDenseX0y2XM0y2X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseX0y1XM0y1X0y1-v3",(None, 400000)),
                                                            ("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 400000))
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        }
                    },
                "gd_agent":
                    {
                        "L111": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDenseX0y0X0y0X0y0-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y5XM0y5X0y2-v3",(None, 200000)),
                                                    ("PandaMyReachDenseX0y3X0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y85XM0y85X0y1-v3",(None, 200000)),
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            #("PandaMyReachDenseX0y4X0y4X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseX0y15X0y25X0y1-v3",(None, 200000)),
                                                            # 5 similar:
                                                            ("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y15XM0y15X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y2XM0y2X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y4XM0y4X0y1-v3",(None, 200000)),
                                                            # 3similar:
                                                            #("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y6XM0y6X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y7XM0y7X0y1-v3",(None, 200000))
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L222": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDenseX0y0X0y0X0y0-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y5XM0y5X0y2-v3",(None, 200000)),
                                                    ("PandaMyReachDenseX0y3X0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y85XM0y85X0y1-v3",(None, 200000)),
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            #("PandaMyReachDenseX0y4X0y4X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseX0y15X0y25X0y1-v3",(None, 200000)),
                                                            # 5 similar:
                                                            ("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y15XM0y15X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y2XM0y2X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y4XM0y4X0y1-v3",(None, 200000)),
                                                            # 3similar:
                                                            #("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y6XM0y6X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y7XM0y7X0y1-v3",(None, 200000))
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L333": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDenseX0y0X0y0X0y0-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y5XM0y5X0y2-v3",(None, 200000)),
                                                    ("PandaMyReachDenseX0y3X0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y85XM0y85X0y1-v3",(None, 200000)),
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PandaMyReachDenseX0y15X0y15X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseX0y15X0y25X0y1-v3",(None, 200000)),
                                                            # 5 similar:
                                                            ("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y15XM0y15X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y2XM0y2X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y4XM0y4X0y1-v3",(None, 200000)),
                                                            # 3similar:
                                                            ("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y6XM0y6X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y7XM0y7X0y1-v3",(None, 200000))
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L444": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDenseX0y0X0y0X0y0-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y5XM0y5X0y2-v3",(None, 200000)),
                                                    ("PandaMyReachDenseX0y3X0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y85XM0y85X0y1-v3",(None, 200000)),
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PandaMyReachDenseX0y15X0y15X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseX0y15X0y25X0y1-v3",(None, 200000)),
                                                            # 5 similar:
                                                            ("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 200000)),
                                                            #("PandaMyReachDenseXM0y15XM0y15X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y2XM0y2X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y4XM0y4X0y1-v3",(None, 200000)),
                                                            # 3similar:
                                                            ("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y6XM0y6X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y7XM0y7X0y1-v3",(None, 200000))
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        },
                        "L555": { # should only be one problem... maybe can add more dynamic goals to show increasing difficulty as number of dynamic goals increase
                            "base_goals_problems_train_configs": [
                                                    ("PandaMyReachDenseX0y0X0y0X0y0-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y3XM0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y5XM0y5X0y2-v3",(None, 200000)),
                                                    ("PandaMyReachDenseX0y3X0y3X0y1-v3",(None, 200000)),
                                                    ("PandaMyReachDenseXM0y85XM0y85X0y1-v3",(None, 200000)),
                                                    ],
                            "dynamic_goals_train_configs": [
                                                            ("PandaMyReachDenseX0y15X0y15X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseX0y2X0y2X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseX0y15X0y25X0y1-v3",(None, 200000)),
                                                            # 5 similar:
                                                            ("PandaMyReachDenseXM0y1XM0y1X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y15XM0y15X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y2XM0y2X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y4XM0y4X0y1-v3",(None, 200000)),
                                                            # 3similar:
                                                            ("PandaMyReachDenseXM0y5XM0y5X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y6XM0y6X0y1-v3",(None, 200000)),
                                                            ("PandaMyReachDenseXM0y7XM0y7X0y1-v3",(None, 200000))
                                                           ],
                            "task_str_to_goal": panda_str_to_goal,
                            "additional_graml_kwargs": {
                                "specified_rl_algorithm_adaptation": SAC, # since this is gc_agent, must be the same as specified_rl_algorithm_learning
                                "input_size": 9,
                                "hidden_size": 8,
                                "batch_size": 32,
                                "num_samples": 20000,
                                "goals_adaptation_sequence_generation_method": GC_AGENT_BASED,
                                "gc_sequence_generation": True,
                                "gc_goal_set": [np.array([sample_goal()]) for i in range(1,30)],
                                "problem_list_to_str_tuple": lambda problems: "_".join([f"[{panda_str_to_goal(s)}]" for s in problems])
                            },
                            "additional_graql_kwargs": {
                                "evaluation_function": mean_wasserstein_distance,
                                "is_universal": True
                            },
                        }
                }
            },
        }
}

# python experiments.py --recognizer graml --domain minigrid --task L111 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L222 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L333 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L444 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L555 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats --inference_same_seq_len

# python experiments.py --recognizer graml --domain minigrid --task L111 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L222 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L333 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L444 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L555 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats --inference_same_seq_len --learn_same_seq_len

# python experiments.py --recognizer graql --domain minigrid --task L111 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L222 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L333 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L444 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L555 --partial_obs_type fragmented --minigrid_env lava_crossing --collect_stats

# python experiments.py --recognizer graql --domain minigrid --task L111 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats 
# python experiments.py --recognizer graql --domain minigrid --task L222 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats 
# python experiments.py --recognizer graql --domain minigrid --task L333 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats 
# python experiments.py --recognizer graql --domain minigrid --task L444 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats 
# python experiments.py --recognizer graql --domain minigrid --task L555 --partial_obs_type continuing --minigrid_env lava_crossing --collect_stats 


# python experiments.py --recognizer graml --domain minigrid --task L1 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L2 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L3 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L4 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L5 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L1 --partial_obs_type continuing --minigrid_env obstacles --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L2 --partial_obs_type continuing --minigrid_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L3 --partial_obs_type continuing --minigrid_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L4 --partial_obs_type continuing --minigrid_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain minigrid --task L5 --partial_obs_type continuing --minigrid_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graql --domain minigrid --task L1 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L2 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L3 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L4 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L5 --partial_obs_type fragmented --minigrid_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L1 --partial_obs_type continuing --minigrid_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L2 --partial_obs_type continuing --minigrid_env obstacles --collect_stats 
# python experiments.py --recognizer graql --domain minigrid --task L3 --partial_obs_type continuing --minigrid_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L4 --partial_obs_type continuing --minigrid_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain minigrid --task L5 --partial_obs_type continuing --minigrid_env obstacles --collect_stats


# python experiments.py --recognizer graml --domain point_maze --task L111 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L222 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L333 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L444 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L555 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L111 --partial_obs_type continuing --point_maze_env obstacles --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L222 --partial_obs_type continuing --point_maze_env obstacles --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L333 --partial_obs_type continuing --point_maze_env obstacles --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L444 --partial_obs_type continuing --point_maze_env obstacles --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L555 --partial_obs_type continuing --point_maze_env obstacles --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graql --domain point_maze --task L111 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L222 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L333 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L444 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L555 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L111 --partial_obs_type continuing --point_maze_env obstacles --collect_stats 
# python experiments.py --recognizer graql --domain point_maze --task L222 --partial_obs_type continuing --point_maze_env obstacles --collect_stats 
# python experiments.py --recognizer graql --domain point_maze --task L333 --partial_obs_type continuing --point_maze_env obstacles --collect_stats 
# python experiments.py --recognizer graql --domain point_maze --task L444 --partial_obs_type continuing --point_maze_env obstacles --collect_stats 
# python experiments.py --recognizer graql --domain point_maze --task L555 --partial_obs_type continuing --point_maze_env obstacles --collect_stats 

# python experiments.py --recognizer graml --domain point_maze --task L111 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L222 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L333 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L444 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L555 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats --inference_same_seq_len

# python experiments.py --recognizer graml --domain point_maze --task L111 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L222 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L333 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L444 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain point_maze --task L555 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats --inference_same_seq_len --learn_same_seq_len

# python experiments.py --recognizer graql --domain point_maze --task L111 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L222 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L333 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L444 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L555 --partial_obs_type fragmented --point_maze_env four_rooms --collect_stats

# python experiments.py --recognizer graql --domain point_maze --task L111 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats 
# python experiments.py --recognizer graql --domain point_maze --task L222 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats 
# python experiments.py --recognizer graql --domain point_maze --task L333 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats 
# python experiments.py --recognizer graql --domain point_maze --task L444 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats 
# python experiments.py --recognizer graql --domain point_maze --task L555 --partial_obs_type continuing --point_maze_env four_rooms --collect_stats 



# python experiments.py --recognizer graml --domain parking --task L111 --partial_obs_type fragmented --parking_env gd_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L222 --partial_obs_type fragmented --parking_env gd_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L333 --partial_obs_type fragmented --parking_env gd_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L444 --partial_obs_type fragmented --parking_env gd_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L555 --partial_obs_type fragmented --parking_env gd_agent --collect_stats --inference_same_seq_len

# python experiments.py --recognizer graml --domain parking --task L111 --partial_obs_type continuing --parking_env gd_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L222 --partial_obs_type continuing --parking_env gd_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L333 --partial_obs_type continuing --parking_env gd_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L444 --partial_obs_type continuing --parking_env gd_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L555 --partial_obs_type continuing --parking_env gd_agent --collect_stats --inference_same_seq_len --learn_same_seq_len

# python experiments.py --recognizer graql --domain parking --task L111 --partial_obs_type fragmented --parking_env gd_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L222 --partial_obs_type fragmented --parking_env gd_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L333 --partial_obs_type fragmented --parking_env gd_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L444 --partial_obs_type fragmented --parking_env gd_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L555 --partial_obs_type fragmented --parking_env gd_agent --collect_stats

# python experiments.py --recognizer graql --domain parking --task L111 --partial_obs_type continuing --parking_env gd_agent --collect_stats 
# python experiments.py --recognizer graql --domain parking --task L222 --partial_obs_type continuing --parking_env gd_agent --collect_stats 
# python experiments.py --recognizer graql --domain parking --task L333 --partial_obs_type continuing --parking_env gd_agent --collect_stats 
# python experiments.py --recognizer graql --domain parking --task L444 --partial_obs_type continuing --parking_env gd_agent --collect_stats 
# python experiments.py --recognizer graql --domain parking --task L555 --partial_obs_type continuing --parking_env gd_agent --collect_stats 


# python experiments.py --recognizer graml --domain parking --task L111 --partial_obs_type fragmented --parking_env gc_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L222 --partial_obs_type fragmented --parking_env gc_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L333 --partial_obs_type fragmented --parking_env gc_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L444 --partial_obs_type fragmented --parking_env gc_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L555 --partial_obs_type fragmented --parking_env gc_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L111 --partial_obs_type continuing --parking_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L222 --partial_obs_type continuing --parking_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L333 --partial_obs_type continuing --parking_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L444 --partial_obs_type continuing --parking_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain parking --task L555 --partial_obs_type continuing --parking_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graql --domain parking --task L111 --partial_obs_type fragmented --parking_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L222 --partial_obs_type fragmented --parking_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L333 --partial_obs_type fragmented --parking_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L444 --partial_obs_type fragmented --parking_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L555 --partial_obs_type fragmented --parking_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L111 --partial_obs_type continuing --parking_env gc_agent --collect_stats 
# python experiments.py --recognizer graql --domain parking --task L222 --partial_obs_type continuing --parking_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain parking --task L333 --partial_obs_type continuing --parking_env gc_agent --collect_stats 
# python experiments.py --recognizer graql --domain parking --task L444 --partial_obs_type continuing --parking_env gc_agent --collect_stats 
# python experiments.py --recognizer graql --domain parking --task L555 --partial_obs_type continuing --parking_env gc_agent --collect_stats 



# python experiments.py --recognizer graml --domain panda --task L111 --partial_obs_type fragmented --panda_env gc_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain panda --task L222 --partial_obs_type fragmented --panda_env gc_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain panda --task L333 --partial_obs_type fragmented --panda_env gc_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain panda --task L444 --partial_obs_type fragmented --panda_env gc_agent --collect_stats --inference_same_seq_len
# python experiments.py --recognizer graml --domain panda --task L555 --partial_obs_type fragmented --panda_env gc_agent --collect_stats --inference_same_seq_len

# python experiments.py --recognizer graml --domain panda --task L111 --partial_obs_type continuing --panda_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain panda --task L222 --partial_obs_type continuing --panda_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain panda --task L333 --partial_obs_type continuing --panda_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain panda --task L444 --partial_obs_type continuing --panda_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len
# python experiments.py --recognizer graml --domain panda --task L555 --partial_obs_type continuing --panda_env gc_agent --collect_stats --inference_same_seq_len --learn_same_seq_len

# python experiments.py --recognizer graql --domain panda --task L111 --partial_obs_type fragmented --panda_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain panda --task L222 --partial_obs_type fragmented --panda_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain panda --task L333 --partial_obs_type fragmented --panda_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain panda --task L444 --partial_obs_type fragmented --panda_env gc_agent --collect_stats
# python experiments.py --recognizer graql --domain panda --task L555 --partial_obs_type fragmented --panda_env gc_agent --collect_stats

# python experiments.py --recognizer graql --domain panda --task L111 --partial_obs_type continuing --panda_env gc_agent --collect_stats 
# python experiments.py --recognizer graql --domain panda --task L222 --partial_obs_type continuing --panda_env gc_agent --collect_stats 
# python experiments.py --recognizer graql --domain panda --task L333 --partial_obs_type continuing --panda_env gc_agent --collect_stats 
# python experiments.py --recognizer graql --domain panda --task L444 --partial_obs_type continuing --panda_env gc_agent --collect_stats 
# python experiments.py --recognizer graql --domain panda --task L555 --partial_obs_type continuing --panda_env gc_agent --collect_stats 
