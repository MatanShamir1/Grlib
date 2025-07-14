from stable_baselines3 import PPO, SAC

from gr_libs import GCAura
from gr_libs.environment._utils.utils import domain_to_env_property
from gr_libs.environment.environment import POINT_MAZE
from gr_libs.metrics import mean_wasserstein_distance, stochastic_amplified_selection
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent
from gr_libs.ml.utils.format import random_subset_with_order


def run_gcaura_point_maze_tutorial():
    """
    Tutorial for GCAura on the Point Maze environment with MultiGoals.

    This tutorial demonstrates:
    1. Training a goal-conditioned model on a goal subspace (center goal only)
    2. Adapting to goals both inside and outside this subspace
    3. Testing inference on all goals types
    """
    print("Starting GCAura tutorial with Point Maze MultiGoals environment...")

    # Use the registered multigoals environment with 3 goals: [(2, 2), (9, 9), (5, 5)]
    # But define our goal subspace to include ONLY the center goal (5, 5)
    # This lets us properly test the subspace functionality

    # Initialize the recognizer with the multigoals empty maze environment
    recognizer = GCAura(
        domain_name=POINT_MAZE,
        env_name="PointMaze-EmptyEnvDense-11x11-MultiGoals-2x2-9x9-5x5",
        evaluation_function=mean_wasserstein_distance,
        finetune_timesteps=70000,  # Fine-tuning for out-of-subspace goals
    )

    # Domain learning phase - train on the center goal subspace only
    print("\nStarting domain learning phase - training on center goal subspace...")
    recognizer.domain_learning_phase(
        {
            "gc": {
                "train_configs": [(SAC, 700000)],
            }
        }
    )

    # Define adaptation goals - one in-subspace (center) and two out-of-subspace (corners)
    # These all exist in the registered environment
    in_subspace_goal = (5, 5)  # Center goal (in subspace)
    out_subspace_goal1 = (9, 1)  # Bottom left corner (out of subspace)
    out_subspace_goal2 = (1, 9)  # Top right corner (out of subspace)

    print(
        "\nStarting goal adaptation phase with both in-subspace and out-of-subspace goals..."
    )

    # Goals adaptation phase with mixed goals
    recognizer.goals_adaptation_phase(
        dynamic_goals=[
            in_subspace_goal,  # In subspace - will use base model
            out_subspace_goal1,  # Out of subspace - will be fine-tuned
            out_subspace_goal2,  # Out of subspace - will be fine-tuned
        ],
    )

    # Setup for testing
    property_type = domain_to_env_property(POINT_MAZE)
    env_property = property_type("PointMaze-EmptyEnvDense-11x11")

    # Create test actor for in-subspace goal (center)
    print("\nCreating test actor for in-subspace goal (center)...")
    problem_name_in = env_property.goal_to_problem_str(in_subspace_goal)
    actor_in = DeepRLAgent(
        domain_name=POINT_MAZE,
        problem_name=problem_name_in,
        env_prop=env_property,
        algorithm=PPO,
        num_timesteps=700000,
    )
    actor_in.learn()

    # Create test actor for out-of-subspace goal (bottom left corner)
    print("\nCreating test actor for out-of-subspace goal (bottom left corner)...")
    problem_name_out = env_property.goal_to_problem_str(out_subspace_goal1)
    actor_out = DeepRLAgent(
        domain_name=POINT_MAZE,
        problem_name=problem_name_out,
        env_prop=env_property,
        algorithm=PPO,
        num_timesteps=500000,
    )
    actor_out.learn()

    # Test inference with in-subspace goal (center)
    print("\nTesting inference with in-subspace goal (should use base model)...")
    full_sequence_in = actor_in.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True,
        with_dict=True,
    )
    partial_sequence_in = random_subset_with_order(
        full_sequence_in, (int)(0.5 * len(full_sequence_in)), is_consecutive=False
    )
    recognized_goal_in = recognizer.inference_phase(
        partial_sequence_in, in_subspace_goal, 0.5
    )
    print(f"Goal recognized for in-subspace sequence: {recognized_goal_in}")
    print(f"Actual goal: {in_subspace_goal}")

    assert str(recognized_goal_in) == str(
        in_subspace_goal
    ), "In-subspace goal recognition failed, expected to recognize the center goal."

    # Test inference with out-of-subspace goal (bottom left corner)
    print(
        "\nTesting inference with out-of-subspace goal (should use fine-tuned model)..."
    )
    full_sequence_out = actor_out.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True,
        with_dict=True,
    )
    partial_sequence_out = random_subset_with_order(
        full_sequence_out, (int)(0.5 * len(full_sequence_out)), is_consecutive=False
    )
    recognized_goal_out = recognizer.inference_phase(
        partial_sequence_out, out_subspace_goal1, 0.5
    )
    print(f"Goal recognized for out-of-subspace sequence: {recognized_goal_out}")
    print(f"Actual goal: {out_subspace_goal1}")

    assert str(recognized_goal_out) == str(
        out_subspace_goal1
    ), "Out-of-subspace goal recognition failed, expected to recognize the bottom left corner goal."

    # Test with second out-of-subspace goal (top right corner)
    print("\nTesting inference with second out-of-subspace goal (top right corner)...")
    problem_name_out2 = env_property.goal_to_problem_str(out_subspace_goal2)
    actor_out2 = DeepRLAgent(
        domain_name=POINT_MAZE,
        problem_name=problem_name_out2,
        env_prop=env_property,
        algorithm=PPO,
        num_timesteps=500000,
    )
    actor_out2.learn()

    full_sequence_out2 = actor_out2.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True,
        with_dict=True,
    )
    partial_sequence_out2 = random_subset_with_order(
        full_sequence_out2, (int)(0.5 * len(full_sequence_out2)), is_consecutive=False
    )
    recognized_goal_out2 = recognizer.inference_phase(
        partial_sequence_out2, out_subspace_goal2, 0.5
    )
    print(
        f"Goal recognized for second out-of-subspace sequence: {recognized_goal_out2}"
    )
    print(f"Actual goal: {out_subspace_goal2}")

    assert str(recognized_goal_out2) == str(
        out_subspace_goal2
    ), "Second out-of-subspace goal recognition failed, expected to recognize the top right corner goal."

    print("\nGCAura Point Maze tutorial completed successfully!")


if __name__ == "__main__":
    run_gcaura_point_maze_tutorial()
