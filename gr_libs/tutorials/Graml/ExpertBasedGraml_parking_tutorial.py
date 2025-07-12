from stable_baselines3 import PPO, SAC, TD3

from gr_libs.environment._utils.utils import domain_to_env_property
from gr_libs.environment.environment import PARKING, ParkingProperty
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent
from gr_libs.ml.utils.format import random_subset_with_order
from gr_libs.recognizer.graml.graml_recognizer import ExpertBasedGraml


def run_ExpertBasedGraml_parking_tutorial():
    recognizer = ExpertBasedGraml(domain_name=PARKING, env_name="Parking-S-14-PC-")

    recognizer.domain_learning_phase(
        {
            "bg": {
                "goals": ["1", "4", "8", "14", "21"],
                "train_configs": [(SAC, 200000) for _ in range(5)],
            }
        }
    )
    recognizer.goals_adaptation_phase(
        dynamic_goals=["1", "11", "21"],
        dynamic_train_configs=[(SAC, 200000) for _ in range(3)],
    )

    property_type = domain_to_env_property(PARKING)
    env_property = property_type("Parking-S-14-PC-")

    # TD3 is different from recognizer and expert algorithms, which are SAC #
    actor = DeepRLAgent(
        domain_name="parking",
        problem_name="Parking-S-14-PC--GI-11-v0",
        env_prop=env_property,
        algorithm=TD3,
        num_timesteps=400000,
    )
    actor.learn()
    # sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
    full_sequence = actor.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True,  # the noise that's added to the actions
    )

    partial_sequence = random_subset_with_order(
        full_sequence, (int)(0.5 * len(full_sequence)), is_consecutive=False
    )
    closest_goal = recognizer.inference_phase(
        partial_sequence,
        ParkingProperty("Parking-S-14-PC--GI-11-v0").str_to_goal(),
        0.5,
    )
    print(
        f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: 11"
    )


if __name__ == "__main__":
    run_ExpertBasedGraml_parking_tutorial()
