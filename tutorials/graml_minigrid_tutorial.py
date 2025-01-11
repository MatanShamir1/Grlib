
from grlib.metrics.metrics import stochastic_amplified_selection
from grlib.ml.tabular.tabular_q_learner import TabularQLearner
from grlib.ml.utils.format import random_subset_with_order
from grlib.recognizer.graml.graml_recognizer import ExpertBasedGraml

# Consider extracting all these to "default point_maze (or every other domain) variables" module which would simplify things like the problem_list_to_str_tuple function, sizes of inputs, etc.
recognizer = ExpertBasedGraml(
    env_name="minigrid", # TODO change to macros which are importable from some info or env module of enums.
    problems=[("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0"),
              ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0"),
              ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0"),
              ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0"),
              ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0"),
              ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0"),
              ("MiniGrid-SimpleCrossingS13N4-DynamicGoal-6x9-v0")],
    method=TabularQLearner,
    collect_statistics=False,
    train_configs=[(None, None) for _ in range(7)],
)
recognizer.domain_learning_phase()
recognizer.goals_adaptation_phase(
    dynamic_goals_problems = ["MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",
                              "MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x11-v0",
                              "MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0",
                              "MiniGrid-SimpleCrossingS13N4-DynamicGoal-7x11-v0",
                              "MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0",
                              "MiniGrid-SimpleCrossingS13N4-DynamicGoal-10x6-v0"],
    dynamic_train_configs=[(None, None) for _ in range(6)] # for expert sequence generation. TODO change to require this only if sequence generation method is EXPERT.
)
# TD3 is different from recognizer and expert algorithms, which are SAC #
actor = TabularQLearner(env_name="minigrid", problem_name="MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0")
actor.learn()
# sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
full_sequence = actor.generate_observation(
    action_selection_method=stochastic_amplified_selection,
    random_optimalism=True, # the noise that's added to the actions
)

partial_sequence = random_subset_with_order(full_sequence, (int)(0.5 * len(full_sequence)), is_consecutive=False)
closest_goal = recognizer.inference_phase(partial_sequence, minigrid_str_to_goal("MiniGrid-SimpleCrossingS13N4-DynamicGoal-8x1-v0"), 0.5)
print(f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: (8, 1)")
