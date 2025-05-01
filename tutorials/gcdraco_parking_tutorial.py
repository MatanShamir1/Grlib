from stable_baselines3 import PPO, TD3, SAC
from gr_libs import GCDraco
from gr_libs.environment.environment import PARKING, ParkingProperty
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent, GCDeepRLAgent
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.utils.format import random_subset_with_order
import gr_envs

def run_gcdraco_parking_tutorial():
    recognizer = GCDraco(
        domain_name=PARKING,
        env_name="Parking-S-14-PC-",
        evaluation_function = "mean_wasserstein_distance",  # or "mean_p_value"
    )
    recognizer.domain_learning_phase(
        [i for i in range(1,21)],
        [(PPO, 200000)]
    )
    recognizer.goals_adaptation_phase(
        dynamic_goals = ["1", "11", "21"]
    )

    actor = DeepRLAgent(domain_name="parking", problem_name="Parking-S-14-PC--GI-11-v0", algorithm=TD3, num_timesteps=400000)
    actor.learn()
    # sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
    full_sequence = actor.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True, # the noise that's added to the actions
        with_dict=True,
    )

    partial_sequence = random_subset_with_order(full_sequence, (int)(0.5 * len(full_sequence)), is_consecutive=False)
    closest_goal = recognizer.inference_phase(partial_sequence, ParkingProperty("Parking-S-14-PC--GI-11-v0").str_to_goal(), 0.5)
    print(f"closest_goal returned by GCDRACO: {closest_goal}\nactual goal actor aimed towards: 11")


if __name__ == "__main__":
    run_gcdraco_parking_tutorial()