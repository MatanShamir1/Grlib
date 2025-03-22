
from stable_baselines3 import PPO, SAC, TD3
from gr_lib.environment.environment import EnvProperty, GCEnvProperty, ParkingProperty
from gr_lib.metrics.metrics import stochastic_amplified_selection
from gr_lib.ml.neural.deep_rl_learner import DeepRLAgent, GCDeepRLAgent
from gr_lib.ml.utils.format import random_subset_with_order
from gr_lib.recognizer.graml.graml_recognizer import ExpertBasedGraml, GCGraml

# Consider extracting all these to "default point_maze (or every other domain) variables" module which would simplify things like the problem_list_to_str_tuple function, sizes of inputs, etc.
recognizer = GCGraml(
    env_name="parking", # TODO change to macros which are importable from some info or env module of enums.
    problems=[ParkingProperty("parking-v0")],
    train_configs=[(PPO, 400000)],
	gc_goal_set=[f"Parking-S-14-PC--GI-{i}-v0" for i in range(1,21)]
)
recognizer.domain_learning_phase()
recognizer.goals_adaptation_phase(
    dynamic_goals_problems = [ParkingProperty(p) for p in ["Parking-S-14-PC--GI-1-v0",
                              "Parking-S-14-PC--GI-4-v0",
                              "Parking-S-14-PC--GI-8-v0",
                              "Parking-S-14-PC--GI-11-v0",
                              "Parking-S-14-PC--GI-14-v0",
                              "Parking-S-14-PC--GI-18-v0",
                              "Parking-S-14-PC--GI-21-v0"]] # TODO detach the goal from the environment instance in every gym env, add the ability to alter it from outside.
    #dynamic_train_configs=[(SAC, 400000) for _ in range(7)] # for expert sequence generation. TODO change to require this only if sequence generation method is EXPERT.
)
# TD3 is different from recognizer and expert algorithms, which are SAC #
actor = DeepRLAgent(env_name="parking", problem_name="Parking-S-14-PC--GI-8-v0", algorithm=TD3, num_timesteps=400000)
actor.learn()
# sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
full_sequence = actor.generate_observation(
    action_selection_method=stochastic_amplified_selection,
    random_optimalism=True, # the noise that's added to the actions
)

partial_sequence = random_subset_with_order(full_sequence, (int)(0.5 * len(full_sequence)), is_consecutive=False)
closest_goal = recognizer.inference_phase(partial_sequence, ParkingProperty("Parking-S-14-PC--GI-8-v0").str_to_goal(), 0.5)
print(f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: 8")
