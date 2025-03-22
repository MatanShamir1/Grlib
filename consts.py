import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from gr_libs.environment.environment import MINIGRID, PARKING, PANDA, POINT_MAZE, QLEARNING, PandaProperty


PROBLEMS = {
	PARKING: {
        "Parking-S-14-PC-": {
			"L1": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [i for i in range(1,21)],
					"train_configs": [(PPO, 200000)]
				},
				"G_0": {
					"goals": ["1", "11", "21"],
					"train_configs": [(SAC, 200000) for _ in range(3)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L2": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [i for i in range(1,21)],
					"train_configs": [(PPO, 200000)]
				},
				"G_0": {
					"goals": ["1", "8", "14", "21"],
					"train_configs": [(SAC, 200000) for _ in range(4)]
				}
			},
			"L3": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [i for i in range(1,21)],
					"train_configs": [(PPO, 200000)]
				},
				"G_0": {
					"goals": ["1", "8", "11", "18"],
					"train_configs": [(SAC, 200000) for _ in range(4)]
				}
			},
			"L4": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [i for i in range(1,21)],
					"train_configs": [(PPO, 200000)]
				},
				"G_0": {
					"goals": ["4", "8", "11", "14", "18"],
					"train_configs": [(SAC, 200000) for _ in range(5)]
				}
			},
			"L5": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [i for i in range(1,21)],
					"train_configs": [(PPO, 200000)]
				},
				"G_0": {
					"goals": ["1", "4", "8", "11", "14", "18", "21"],
					"train_configs": [(SAC, 200000) for _ in range(7)]
				}
			}
		}
    }, # PARKING
	PANDA: {
	"PandaMyReachDense": {
			"L1": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [np.array([PandaProperty.sample_goal()]) for _ in range(1,30)],
					"train_configs": [(SAC, 800000)]
				},
				"G_0": {
					"goals": [np.array([[-0.1, -0.1, 0.1]]), np.array([[-0.1, 0.1, 0.1]]), np.array([[0.2, 0.2, 0.1]])],
					"train_configs": [(SAC, 00000), (SAC, 200000), (SAC, 300000)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L2": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [np.array([PandaProperty.sample_goal()]) for _ in range(1,30)],
					"train_configs": [(SAC, 800000)]
				},
				"G_0": {
					"goals": [np.array([[-0.5, -0.5, 0.1]]), np.array([[-0.5, 0.2, 0.1]]), np.array([[-0.1, 0.1, 0.1]]), np.array([[0.1, -0.1, 0.1]])],
					"train_configs": [(SAC, 400000), (SAC, 400000), (SAC, 400000), (SAC, 400000)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L3": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [np.array([PandaProperty.sample_goal()]) for _ in range(1,30)],
					"train_configs": [(SAC, 800000)]
				},
				"G_0": {
					"goals": [np.array([[-0.5, -0.5, 0.1]]), np.array([[-0.1, -0.1, 0.1]]), np.array([[-0.5, 0.2, 0.1]]), np.array([[-0.1, 0.1, 0.1]]), np.array([[0.2, -0.2, 0.1]]), np.array([[0.2, 0.2, 0.1]])],
					"train_configs": [(SAC, 400000) for _ in range(6)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L4": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [np.array([PandaProperty.sample_goal()]) for _ in range(1,30)],
					"train_configs": [(SAC, 800000)]
				},
				"G_0": {
					"goals": [np.array([[-0.3, -0.3, 0.1]]), np.array([[-0.1, -0.1, 0.1]]), np.array([[-0.3, 0.2, 0.1]]), np.array([[-0.1, 0.1, 0.1]]), np.array([[0.1, -0.1, 0.1]]), np.array([[0.2, 0.2, 0.1]])],
					"train_configs": [(SAC, 400000) for _ in range(6)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L5": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [np.array([PandaProperty.sample_goal()]) for _ in range(1,30)],
					"train_configs": [(SAC, 800000)]
				},
				"G_0": {
					"goals": [np.array([[-0.5, -0.5, 0.1]]), np.array([[-0.3, -0.3, 0.1]]), np.array([[-0.1, -0.1, 0.1]]), np.array([[-0.5, 0.2, 0.1]]), np.array([[-0.3, 0.2, 0.1]]), np.array([[-0.1, 0.1, 0.1]]), np.array([[0.2, -0.2, 0.1]]), np.array([[0.1, -0.1, 0.1]]), np.array([[0.2, 0.2, 0.1]])],
					"train_configs": [(SAC, 400000) for _ in range(9)] # algorithms that use GC agent to generate sequences don't use this
				}
			}
		}
    }, # PANDA
	POINT_MAZE: {
        "PointMaze-FourRoomsEnvDense-11x11": {
			"L1": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(9,1), (9,9), (1,9), (3,3), (3,4), (8,2), (3,7), (2,8)],
					"train_configs": [(SAC, 400000) for _ in range(8)]
				},
				"G_0": {
					"goals": [(4,4), (7,3), (3,7)],
					"train_configs": [(SAC, 400000) for _ in range(3)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L2": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(9,1), (9,9), (1,9), (3,3), (3,4), (8,2), (3,7), (2,8)],
					"train_configs": [(SAC, 400000) for _ in range(8)]
				},
				"G_0": {
					"goals": [(4,4), (7,3), (3,7), (8,2)],
					"train_configs": [(SAC, 400000) for _ in range(4)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L3": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(9,1), (9,9), (1,9), (3,3), (3,4), (8,2), (3,7), (2,8)],
					"train_configs": [(SAC, 400000) for _ in range(8)]
				},
				"G_0": {
					"goals": [(4,4), (7,3), (3,7), (8,2), (2,8)],
					"train_configs": [(SAC, 400000) for _ in range(5)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L4": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(9,1), (9,9), (1,9), (3,3), (3,4), (8,2), (3,7), (2,8)],
					"train_configs": [(SAC, 400000) for _ in range(8)]
				},
				"G_0": {
					"goals": [(4,4), (7,3), (3,7), (8,2), (2,8), (3,4)],
					"train_configs": [(SAC, 400000) for _ in range(6)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L5": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(9,1), (9,9), (1,9), (3,3), (3,4), (8,2), (3,7), (2,8)],
					"train_configs": [(SAC, 400000) for _ in range(8)]
				},
				"G_0": {
					"goals": [(4,4), (7,3), (3,7), (8,2), (2,8), (3,4), (4,3)],
					"train_configs": [(SAC, 400000) for _ in range(7)] # algorithms that use GC agent to generate sequences don't use this
				}
			}
		},
		"PointMaze-ObstaclesEnvDense-11x11": {
			"L1": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(5,1), (9,9), (1,5), (6,4), (4,6), (6,6), (7,7)],
					"train_configs": [(SAC, 400000) for _ in range(7)]
				},
				"G_0": {
					"goals": [(5,5), (7,4), (4,7)],
					"train_configs": [(SAC, 400000) for _ in range(3)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L2": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(5,1), (9,9), (1,5), (6,4), (4,6), (6,6), (7,7)],
					"train_configs": [(SAC, 400000) for _ in range(7)]
				},
				"G_0": {
					"goals": [(5,5), (3,6), (7,4)],
					"train_configs": [(SAC, 400000) for _ in range(3)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L3": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(5,1), (9,9), (1,5), (6,4), (4,6), (6,6), (7,7)],
					"train_configs": [(SAC, 400000) for _ in range(7)]
				},
				"G_0": {
					"goals": [(5,5), (3,6), (7,4), (4,7)],
					"train_configs": [(SAC, 400000) for _ in range(4)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L4": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(5,1), (9,9), (1,5), (6,4), (4,6), (6,6), (7,7)],
					"train_configs": [(SAC, 400000) for _ in range(7)]
				},
				"G_0": {
					"goals": [(3,6), (6,3), (7,4), (4,7), (8,8)],
					"train_configs": [(SAC, 400000) for _ in range(5)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L5": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(5,1), (9,9), (1,5), (6,4), (4,6), (6,6), (7,7)],
					"train_configs": [(SAC, 400000) for _ in range(7)]
				},
				"G_0": {
					"goals": [(5,5), (3,6), (6,3), (7,4), (4,7), (8,8)],
					"train_configs": [(SAC, 400000) for _ in range(6)] # algorithms that use GC agent to generate sequences don't use this
				}
			}
		}
    }, # POINT_MAZE
	MINIGRID: {
        "MiniGrid-SimpleCrossingS13N4": {
			"L1": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(11,1), (11,11), (1,11), (7,11), (8,1), (10,6), (6,9), (11,3), (11,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(9)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(11,1), (11,11), (1,11)],
					"train_configs": [(QLEARNING, 100000) for _ in range(3)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L2": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(11,1), (11,11), (1,11), (7,11), (8,1), (10,6), (6,9), (11,3), (11,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(9)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(11,1), (11,11), (1,11), (5,9)],
					"train_configs": [(QLEARNING, 100000) for _ in range(4)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L3": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(11,1), (11,11), (1,11), (7,11), (8,1), (10,6), (6,9), (11,3), (11,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(9)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(11,1), (11,11), (1,11), (5,9), (6,1)],
					"train_configs": [(QLEARNING, 100000) for _ in range(5)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L4": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(11,1), (11,11), (1,11), (7,11), (8,1), (10,6), (6,9), (11,3), (11,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(9)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(11,1), (11,11), (1,11), (5,9), (6,1), (11,3)],
					"train_configs": [(QLEARNING, 100000) for _ in range(6)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L5": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(11,1), (11,11), (1,11), (7,11), (8,1), (10,6), (6,9), (11,3), (11,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(9)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(11,1), (11,11), (1,11), (5,9), (6,1), (11,3), (11,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(7)] # algorithms that use GC agent to generate sequences don't use this
				}
			}
		},
		"MiniGrid-LavaCrossingS9N2": {
			"L1": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(7,7), (1,7), (7,1), (1,3), (2,5), (5,2), (6,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(7)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(1,3), (6,5), (4,7)],
					"train_configs": [(QLEARNING, 100000) for _ in range(3)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L2": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(7,7), (1,7), (7,1), (1,3), (2,5), (5,2), (6,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(7)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(1,3), (6,5), (4,7), (2,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(4)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L3": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(7,7), (1,7), (7,1), (1,3), (2,5), (5,2), (6,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(7)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(1,3), (6,5), (4,7), (2,5), (5,2)],
					"train_configs": [(QLEARNING, 100000) for _ in range(5)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L4": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(7,7), (1,7), (7,1), (1,3), (2,5), (5,2), (6,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(7)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(1,3), (6,5), (4,7), (2,5), (5,2), (4,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(6)] # algorithms that use GC agent to generate sequences don't use this
				}
			},
			"L5": {
				"base": { # only 1 train_config for the GC agent. the goals are samples of possible goals in the env.
					"goals": [(7,7), (1,7), (7,1), (1,3), (2,5), (5,2), (6,5)],
					"train_configs": [(QLEARNING, 100000) for _ in range(7)] # TODO Find a way to prevent the 'none' or simply accept only a Q-learning algorithm and assert everything else
				},
				"G_0": {
					"goals": [(1,3), (6,5), (4,7), (2,5), (5,2), (4,5), (1,1)],
					"train_configs": [(QLEARNING, 100000) for _ in range(7)] # algorithms that use GC agent to generate sequences don't use this
				}
			}
		}
    }, # MINIGRID
} # PROBLEMS

for i,perc in enumerate([0.3, 0.5, 0.7, 0.9, 1]):
	for j, cons in enumerate([True, False]):
		
		### PARKING ###

		PROBLEMS[PARKING]["Parking-S-14-PC-"]["L1"].update(
				{f"I_0_{i*6+j*3}": {
					"goal": "1",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+1}": {
					"goal": "11",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+2}": {
					"goal": "21",
					"train_config": (TD3, 300000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[PARKING]["Parking-S-14-PC-"]["L2"].update(
				{f"I_0_{i*8+j*4}": {
					"goal": "1",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+1}": {
					"goal": "8",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+2}": {
					"goal": "14",
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+3}": {
					"goal": "21",
					"train_config": (TD3, 300000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[PARKING]["Parking-S-14-PC-"]["L3"].update(
				{f"I_0_{i*8+j*4}": {
					"goal": "1",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+1}": {
					"goal": "8",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+2}": {
					"goal": "11",
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+3}": {
					"goal": "18",
					"train_config": (TD3, 300000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[PARKING]["Parking-S-14-PC-"]["L4"].update(
				{f"I_0_{i*10+j*5}": {
					"goal": "4",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+1}": {
					"goal": "8",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+2}": {
					"goal": "11",
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+3}": {
					"goal": "14",
					"train_config": (TD3, 300000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+4}": {
					"goal": "18",
					"train_config": (TD3, 300000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[PARKING]["Parking-S-14-PC-"]["L5"].update(
				{f"I_0_{i*14+j*7}": {
					"goal": "1",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+1}": {
					"goal": "4",
					"train_config": (TD3, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+2}": {
					"goal": "8",
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+3}": {
					"goal": "11",
					"train_config": (TD3, 300000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+4}": {
					"goal": "14",
					"train_config": (TD3, 300000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+5}": {
					"goal": "18",
					"train_config": (TD3, 300000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+6}": {
					"goal": "21",
					"train_config": (TD3, 300000),
					"consecutive": cons,
					"percentage": perc
				}})

		### PANDA ###

		PROBLEMS[PANDA]["PandaMyReachDense"]["L1"].update(
				{f"I_0_{i*6+j*3}": {
					"goal": np.array([[-0.1, -0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+1}": {
					"goal": np.array([[-0.1, 0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+2}": {
					"goal": np.array([[0.2, 0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[PANDA]["PandaMyReachDense"]["L2"].update(
				{f"I_0_{i*8+j*4}": {
					"goal": np.array([[-0.5, -0.5, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+1}": {
					"goal": np.array([[-0.5, 0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+2}": {
					"goal": np.array([[-0.1, 0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+3}": {
					"goal": np.array([[0.1, -0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[PANDA]["PandaMyReachDense"]["L3"].update(
				{f"I_0_{i*12+j*6}": {
					"goal": np.array([[-0.5, -0.5, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+1}": {
					"goal": np.array([[-0.1, -0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+2}": {
					"goal": np.array([[-0.5, 0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+3}": {
					"goal": np.array([[-0.1, 0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+4}": {
					"goal": np.array([[0.2, -0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+5}": {
					"goal": np.array([[0.2, 0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[PANDA]["PandaMyReachDense"]["L4"].update(
				{f"I_0_{i*12+j*6}": {
					"goal": np.array([[-0.3, -0.3, 0.1]]),
					"train_config": (SAC, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+1}": {
					"goal": np.array([[-0.1, -0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+2}": {
					"goal": np.array([[-0.3, 0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+3}": {
					"goal": np.array([[-0.1, 0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+4}": {
					"goal": np.array([[0.1, -0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+5}": {
					"goal": np.array([[0.2, 0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[PANDA]["PandaMyReachDense"]["L5"].update(
				{f"I_0_{i*18+j*9}": {
					"goal": np.array([[-0.5, -0.5, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*18+j*9+1}": {
					"goal": np.array([[-0.3, -0.3, 0.1]]),
					"train_config": (SAC, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*18+j*9+2}": {
					"goal": np.array([[-0.1, -0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*18+j*9+3}": {
					"goal": np.array([[-0.5, 0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*18+j*9+4}": {
					"goal": np.array([[-0.3, 0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*18+j*9+5}": {
					"goal": np.array([[-0.1, 0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*18+j*9+6}": {
					"goal": np.array([[0.2, -0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*18+j*9+7}": {
					"goal": np.array([[0.1, -0.1, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*18+j*9+8}": {
					"goal": np.array([[0.2, 0.2, 0.1]]),
					"train_config": (PPO, 200000),
					"consecutive": cons,
					"percentage": perc
				}})
		
		### POINT_MAZE ###

		PROBLEMS[POINT_MAZE]["PointMaze-FourRoomsEnvDense-11x11"]["L1"].update( # TODO the existing working 9x9 is not Dense. need to duplicate it for the dense one
				{f"I_0_{i*6+j*3}": {
					"goal": (4,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+1}": {
					"goal": (7,3),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+2}": {
					"goal": (3,7),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[POINT_MAZE]["PointMaze-FourRoomsEnvDense-11x11"]["L2"].update(
				{f"I_0_{i*8+j*4}": {
					"goal": (4,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+1}": {
					"goal": (7,3),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+2}": {
					"goal": (3,7),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+3}": {
					"goal": (8,2),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[POINT_MAZE]["PointMaze-FourRoomsEnvDense-11x11"]["L3"].update(
				{f"I_0_{i*10+j*5}": {
					"goal": (4,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+1}": {
					"goal": (7,3),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+2}": {
					"goal": (3,7),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+3}": {
					"goal": (8,2),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+4}": {
					"goal": (2,8),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[POINT_MAZE]["PointMaze-FourRoomsEnvDense-11x11"]["L4"].update(
				{f"I_0_{i*12+j*6}": {
					"goal": (4,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+1}": {
					"goal": (7,3),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+2}": {
					"goal": (3,7),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+3}": {
					"goal": (8,2),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+4}": {
					"goal": (2,8),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+5}": {
					"goal": (3,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[POINT_MAZE]["PointMaze-FourRoomsEnvDense-11x11"]["L5"].update(
				{f"I_0_{i*14+j*7}": {
					"goal": (4,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+1}": {
					"goal": (7,3),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+2}": {
					"goal": (3,7),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+3}": {
					"goal": (8,2),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+4}": {
					"goal": (2,8),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+5}": {
					"goal": (3,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+6}": {
					"goal": (4,3),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})

		PROBLEMS[POINT_MAZE]["PointMaze-ObstaclesEnvDense-11x11"]["L1"].update(
				{f"I_0_{i*6+j*3}": {
					"goal": (5,5),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+1}": {
					"goal": (7,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+2}": {
					"goal": (4,7),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[POINT_MAZE]["PointMaze-ObstaclesEnvDense-11x11"]["L2"].update(
				{f"I_0_{i*6+j*3}": {
					"goal": (5,5),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+1}": {
					"goal": (3,6),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+2}": {
					"goal": (7,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[POINT_MAZE]["PointMaze-ObstaclesEnvDense-11x11"]["L3"].update(
				{f"I_0_{i*8+j*4}": {
					"goal": (5,5),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+1}": {
					"goal": (3,6),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+2}": {
					"goal": (7,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+3}": {
					"goal": (4,7),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[POINT_MAZE]["PointMaze-ObstaclesEnvDense-11x11"]["L4"].update(
				{f"I_0_{i*10+j*5}": {
					"goal": (5,5),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+1}": {
					"goal": (3,6),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+2}": {
					"goal": (7,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+3}": {
					"goal": (4,7),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+4}": {
					"goal": (8,8),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[POINT_MAZE]["PointMaze-ObstaclesEnvDense-11x11"]["L5"].update(
				{f"I_0_{i*12+j*6}": {
					"goal": (5,5),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+1}": {
					"goal": (3,6),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+2}": {
					"goal": (6,3),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+3}": {
					"goal": (7,4),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+4}": {
					"goal": (4,7),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+5}": {
					"goal": (8,8),
					"train_config": (TD3, 400000),
					"consecutive": cons,
					"percentage": perc
				}})

		### MINIGRID ###

		PROBLEMS[MINIGRID]["MiniGrid-SimpleCrossingS13N4"]["L1"].update(
				{f"I_0_{i*6+j*3}": {
					"goal": (11,1),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+1}": {
					"goal": (1,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+2}": {
					"goal": (11,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[MINIGRID]["MiniGrid-SimpleCrossingS13N4"]["L2"].update(
				{f"I_0_{i*8+j*4}": {
					"goal": (11,1),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+1}": {
					"goal": (1,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+2}": {
					"goal": (11,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+3}": {
					"goal": (5,9),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[MINIGRID]["MiniGrid-SimpleCrossingS13N4"]["L3"].update(
				{f"I_0_{i*10+j*5}": {
					"goal": (11,1),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+1}": {
					"goal": (1,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+2}": {
					"goal": (11,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+3}": {
					"goal": (5,9),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+4}": {
					"goal": (6,1),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[MINIGRID]["MiniGrid-SimpleCrossingS13N4"]["L4"].update(
				{f"I_0_{i*12+j*6}": {
					"goal": (11,1),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+1}": {
					"goal": (1,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+2}": {
					"goal": (11,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+3}": {
					"goal": (5,9),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+5}": {
					"goal": (11,3),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[MINIGRID]["MiniGrid-SimpleCrossingS13N4"]["L5"].update(
				{f"I_0_{i*14+j*7}": {
					"goal": (11,1),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+1}": {
					"goal": (1,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+2}": {
					"goal": (11,11),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+3}": {
					"goal": (5,9),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+5}": {
					"goal": (11,3),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+6}": {
					"goal": (11,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})

		PROBLEMS[MINIGRID]["MiniGrid-LavaCrossingS9N2"]["L1"].update(
				{f"I_0_{i*6+j*3}": {
					"goal": (1,3),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+1}": {
					"goal": (6,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*6+j*3+2}": {
					"goal": (4,7),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[MINIGRID]["MiniGrid-LavaCrossingS9N2"]["L2"].update(
				{f"I_0_{i*8+j*4}": {
					"goal": (1,3),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+1}": {
					"goal": (6,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+2}": {
					"goal": (4,7),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*8+j*4+3}": {
					"goal": (2,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[MINIGRID]["MiniGrid-LavaCrossingS9N2"]["L3"].update(
				{f"I_0_{i*10+j*5}": {
					"goal": (1,3),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+1}": {
					"goal": (6,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+2}": {
					"goal": (4,7),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+3}": {
					"goal": (2,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*10+j*5+4}": {
					"goal": (5,2),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[MINIGRID]["MiniGrid-LavaCrossingS9N2"]["L4"].update(
				{f"I_0_{i*12+j*6}": {
					"goal": (1,3),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+1}": {
					"goal": (6,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+2}": {
					"goal": (4,7),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+3}": {
					"goal": (2,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+4}": {
					"goal": (5,2),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*12+j*6+5}": {
					"goal": (4,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})
		PROBLEMS[MINIGRID]["MiniGrid-LavaCrossingS9N2"]["L5"].update(
				{f"I_0_{i*14+j*7}": {
					"goal": (1,3),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+1}": {
					"goal": (6,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+2}": {
					"goal": (4,7),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+3}": {
					"goal": (2,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+4}": {
					"goal": (5,2),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+5}": {
					"goal": (4,5),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				},
				f"I_0_{i*14+j*7+6}": {
					"goal": (1,1),
					"train_config": (QLEARNING, 100000),
					"consecutive": cons,
					"percentage": perc
				}})
