
from stable_baselines3 import SAC, TD3
from gr_lib.environment.utils.format import maze_str_to_goal
from gr_lib.metrics.metrics import stochastic_amplified_selection
from gr_lib.ml.neural.deep_rl_learner import DeepRLAgent
from gr_lib.ml.utils.format import random_subset_with_order
from gr_lib.recognizer.graml.graml_recognizer import ExpertBasedGraml

# Consider extracting all these to "default point_maze (or every other domain) variables" module which would simplify things like the problem_list_to_str_tuple function, sizes of inputs, etc.
recognizer = ExpertBasedGraml(
    env_name="point_maze", # TODO change to macros which are importable from some info or env module of enums.
    problems=[("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1"),
              ("PointMaze-FourRoomsEnv-11x11-Goal-9x9"), # this one doesn't work with dense rewards because of encountering local minima
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8")],
    task_str_to_goal=maze_str_to_goal,
    method=DeepRLAgent,
    collect_statistics=False,
    train_configs=[(SAC, 200000) for _ in range(8)],
)
recognizer.domain_learning_phase()
recognizer.goals_adaptation_phase(
    dynamic_goals_problems = ["PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",
                              "PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",
                              "PointMaze-FourRoomsEnvDense-11x11-Goal-3x7"],
    dynamic_train_configs=[(SAC, 200000) for _ in range(3)] # for expert sequence generation. TODO change to require this only if sequence generation method is EXPERT.
)
# TD3 is different from recognizer and expert algorithms, which are SAC #
actor = DeepRLAgent(env_name="point_maze", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-4x4", algorithm=TD3, num_timesteps=200000)
actor.learn()
# sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
full_sequence = actor.generate_observation(
    action_selection_method=stochastic_amplified_selection,
    random_optimalism=True, # the noise that's added to the actions
)

partial_sequence = random_subset_with_order(full_sequence, (int)(0.5 * len(full_sequence)))
closest_goal = recognizer.inference_phase(partial_sequence, maze_str_to_goal("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4"), 0.5)
print(f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: (4, 4)")
