import numpy as np
from stable_baselines3 import PPO, SAC

from gr_libs import GCAura
from gr_libs.environment._utils.utils import domain_to_env_property
from gr_libs.environment.environment import PANDA
from gr_libs.metrics import mean_wasserstein_distance, stochastic_amplified_selection
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent
from gr_libs.ml.utils.format import random_subset_with_order


def run_gcaura_panda_tutorial():
    """
    Tutorial for GCAura on the Panda environment.

    This tutorial demonstrates:
    1. Training a goal-conditioned model on a registered goal subspace (center area)
    2. Adapting to goals both inside and outside this subspace
    3. Testing inference on multiple goal types
    """
    print("Starting GCAura tutorial with Panda environment...")

    print(f"Using training subspace with center-area goals")

    # Initialize the recognizer with the center subspace environment
    recognizer = GCAura(
        domain_name=PANDA,
        env_name="PandaMyReachDenseSubspaceCenterOnly-v3",  # Use the subspace environment
        evaluation_function=mean_wasserstein_distance,
        finetune_timesteps=50000,
    )

    # Domain learning phase - train on the center goal subspace
    print("\nStarting domain learning phase - training on registered goal subspace...")
    recognizer.domain_learning_phase(
        {
            "gc": {
                "train_configs": [(SAC, 500000)],
            }
        }
    )

    # Define adaptation goals - mix of in-subspace and out-of-subspace goals
    # Use predefined goals from our environment registration
    in_subspace_goal = np.array([[0.0, 0.0, 0.1]])  # Center goal (in subspace)
    out_subspace_goal1 = np.array([[-0.3, -0.3, 0.1]])  # Far corner (out of subspace)
    out_subspace_goal2 = np.array([[0.2, 0.2, 0.1]])  # Far corner (out of subspace)

    print(
        "\nStarting goal adaptation phase with both in-subspace and out-of-subspace goals..."
    )

    # Goals adaptation phase with mixed goals
    recognizer.goals_adaptation_phase(
        dynamic_goals=[
            in_subspace_goal,
            out_subspace_goal1,
            out_subspace_goal2,
        ],
    )

    # Setup for testing
    property_type = domain_to_env_property(PANDA)
    env_property = property_type("PandaMyReachDense")

    # Create test actor for in-subspace goal
    print("\nCreating test actor for in-subspace goal...")
    problem_name_in = env_property.goal_to_problem_str(in_subspace_goal)
    actor_in = DeepRLAgent(
        domain_name=PANDA,
        problem_name=problem_name_in,
        env_prop=env_property,
        algorithm=PPO,
        num_timesteps=250000,
    )
    actor_in.learn()

    # Create test actor for out-of-subspace goal
    print("\nCreating test actor for out-of-subspace goal...")
    problem_name_out = env_property.goal_to_problem_str(out_subspace_goal1)
    actor_out = DeepRLAgent(
        domain_name=PANDA,
        problem_name=problem_name_out,
        env_prop=env_property,
        algorithm=PPO,
        num_timesteps=250000,
    )
    actor_out.learn()

    # Test inference with in-subspace goal
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
    ), f"In-subspace goal recognition failed. Expected goal does not match recognized goal {recognized_goal_in}."

    # Test inference with out-of-subspace goal
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
    ), f"Out-of-subspace goal recognition failed. Expected goal does not match recognized goal {recognized_goal_out}."

    # Try another out-of-subspace goal
    print("\nTesting inference with second out-of-subspace goal...")
    problem_name_out2 = env_property.goal_to_problem_str(out_subspace_goal2)
    actor_out2 = DeepRLAgent(
        domain_name=PANDA,
        problem_name=problem_name_out2,
        env_prop=env_property,
        algorithm=PPO,
        num_timesteps=250000,
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
    ), f"Out-of-subspace goal recognition failed. Expected goal does not match recognized goal {recognized_goal_out2}."

    print("\nGCAura tutorial completed successfully!")


if __name__ == "__main__":
    run_gcaura_panda_tutorial()
