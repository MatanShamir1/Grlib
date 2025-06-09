import numpy as np
from stable_baselines3 import PPO, SAC

from gr_libs import GCAura
from gr_envs.maze_scripts.envs.maze.generate_maze import gen_empty_env
from gr_libs.environment.environment import POINT_MAZE, PointMazeProperty
from gr_libs.metrics import mean_wasserstein_distance, stochastic_amplified_selection
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent
from gr_libs.ml.utils.format import random_subset_with_order


def run_gcaura_pointmaze_tutorial():
    """
    End-to-end tutorial for GCAura on a PointMaze FourRooms environment.
    1. Dynamically register and wrap the base env into an Aura env with a goal sub-space.
    2. Train a base goal-conditioned policy over the goal sub-space.
    3. Adapt to new goals—reusing the base policy for in-space goals, fine-tuning for outside goals.
    4. Generate a test trajectory and perform inference.
    """

    # 1. Define your goal sub-space: the four corners of the 11×11 grid
    goal_subspace = [(9, 1), (1, 9), (9, 9)]

    # 2. Instantiate GCAura; it will register "<env_name>-Aura" behind the scenes
    recognizer = GCAura(
        domain_name=POINT_MAZE,
        env_name="PointMaze-EmptyEnvDense-11x11",
        gc_entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
        gc_kwargs={
            # these kwargs are passed straight to the constructor of PointMazeGCEnv
            "reward_type": "dense",
            "maze_map": gen_empty_env(11, 11, [(1, 1)], goal_subspace)
        },
        gc_max_steps=900,
        evaluation_function=mean_wasserstein_distance,
    )

    # 3. Domain learning: train over the goal sub-space itself
    base_goals = [np.array(g) for g in goal_subspace]
    base_train_configs = [(SAC, 400_0)]
    recognizer.domain_learning_phase(
        base_goals=base_goals,
        train_configs=base_train_configs,
    )

    # 4. Goal adaptation: mix in-space and out-of-space goals
    dynamic_goals = [
        np.array((9, 9)),  # inside goal sub-space -> reuse base policy
        np.array((4, 4)),  # outside -> fine-tune
        np.array((8, 8)),  # outside -> fine-tune
    ]
    adaptation_configs = [
        (),  # empty tuple is ignored for in-space goals
        (SAC, 400_0),  # 100k timesteps fine-tuning for goal (4,4)
        (SAC, 400_0),  # 100k timesteps fine-tuning for goal (8,8)
    ]
    recognizer.goals_adaptation_phase(
        dynamic_goals=dynamic_goals,
        adaptation_configs=adaptation_configs,
    )

    # 5. Prepare an expert actor for one of the new goals (e.g. (4,4))
    aura_env_id = "PointMaze-FourRoomsEnvDense-11x11-Aura"
    # Instantiate a matching EnvProperty (with the same goal_subspace)
    env_prop = PointMazeProperty(aura_env_id, goal_subspace=goal_subspace)
    problem_name = env_prop.goal_to_problem_str((4, 4))
    actor = DeepRLAgent(
        domain_name=POINT_MAZE,
        problem_name=problem_name,
        env_prop=env_prop,
        algorithm=PPO,
        # num_timesteps=200_000,
        num_timesteps=400_0
    )
    actor.learn()

    # 6. Generate a full trajectory, then take a 50% random subset
    full_seq = actor.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True,
        with_dict=True,
    )
    partial_seq = random_subset_with_order(
        full_seq,
        int(0.5 * len(full_seq)),
        is_consecutive=False,
    )

    # 7. Inference: which goal does GCAura pick?
    closest = recognizer.inference_phase(
        partial_seq,
        np.array((4, 4)),
        0.5,
    )
    print(f"closest_goal returned by GCAura: {closest}\nactual goal: (4, 4)")


if __name__ == "__main__":
    run_gcaura_pointmaze_tutorial()
