import numpy as np
from stable_baselines3 import PPO

from gr_libs import Draco
from gr_libs.environment._utils.utils import domain_to_env_property
from gr_libs.environment.environment import PANDA
from gr_libs.metrics import mean_wasserstein_distance
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent
from gr_libs.ml.utils.format import random_subset_with_order


def run_draco_panda_tutorial():
    recognizer = Draco(
        domain_name=PANDA,
        env_name="PandaMyReachDense",
        evaluation_function=mean_wasserstein_distance,  # or mean_p_value
    )

    recognizer.goals_adaptation_phase(
        dynamic_goals=[
            np.array([[-0.1, -0.1, 0.1]]),
            np.array([[-0.1, 0.1, 0.1]]),
            np.array([[0.2, 0.2, 0.1]]),
        ],
        dynamic_train_configs=[(PPO, 200000), (PPO, 200000), (PPO, 200000)],
    )
    # TD3 is different from recognizer and expert algorithms, which are SAC #
    property_type = domain_to_env_property(PANDA)
    env_property = property_type("PandaMyReachDense")
    problem_name = env_property.goal_to_problem_str(np.array([[-0.1, -0.1, 0.1]]))
    actor = DeepRLAgent(
        domain_name=PANDA,
        problem_name=problem_name,
        env_prop=env_property,
        algorithm=PPO,
        num_timesteps=400000,
    )
    actor.learn()
    # sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
    full_sequence = actor.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True,  # the noise that's added to the actions
        with_dict=True,
    )
    partial_sequence = random_subset_with_order(
        full_sequence, (int)(0.5 * len(full_sequence)), is_consecutive=False
    )
    closest_goal = recognizer.inference_phase(
        partial_sequence, np.array([[-0.1, -0.1, 0.1]]), 0.5
    )
    print(
        f"closest_goal returned by DRACO: {closest_goal}\nactual goal actor aimed towards: [-0.1, -0.1, 0.1]"
    )


if __name__ == "__main__":
    run_draco_panda_tutorial()
