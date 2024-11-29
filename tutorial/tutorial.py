
from stable_baselines3 import SAC, TD3
from grlib.environment.utils.format import maze_str_to_goal
from grlib.metrics.metrics import stochastic_amplified_selection
from grlib.ml.neural.SB3model import NeuralAgent
from grlib.ml.utils.format import random_subset_with_order
from grlib.ml.utils.storage import set_global_storage_configs
from grlib.recognizer.graml.graml_recognizer import GramlRecognizer
from grlib.recognizer.graml.graml_recognizer import AGENT_BASED

set_global_storage_configs(recognizer_str="graml", is_fragmented="fragmented", is_inference_same_length_sequences=True, is_learn_same_length_sequences=False) # TODO instead of setting the global storage config, do it from the recognizer directly

# Consider extracting all these to "default point_maze (or every other domain) variables" module which would simplify things like the problem_list_to_str_tuple function, sizes of inputs, etc.
recognizer = GramlRecognizer(
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
    method=NeuralAgent,
    collect_statistics=False,
    train_configs=[(SAC, 200000) for i in range(8)],
    partial_obs_type="fragmented",
    batch_size=32,
    input_size=6,
    hidden_size=8,
    num_samples=20000,
    problem_list_to_str_tuple=lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems]),
    is_learn_same_length_sequences=False,
    goals_adaptation_sequence_generation_method=AGENT_BASED # take expert samples in goals adaptation phase
)
recognizer.domain_learning_phase()
recognizer.goals_adaptation_phase(
    dynamic_goals_problems = ["PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",
                              "PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",
                              "PointMaze-FourRoomsEnvDense-11x11-Goal-3x7"],
    dynamic_train_configs=[(SAC, 200000) for i in range(3)] # for expert sequence generation. TODO change to require this only if sequence generation method is EXPERT.
)
# TD3 is different from recognizer and expert algorithms, which are SAC #
actor = NeuralAgent(env_name="point_maze", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-4x4", algorithm=TD3, num_timesteps=200000)
actor.learn()
# sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
full_sequence = actor.generate_observation(
    action_selection_method=stochastic_amplified_selection,
    random_optimalism=True, # the noise that's added to the actions
)

partial_sequence = random_subset_with_order(full_sequence, (int)(0.5 * len(full_sequence)), is_fragmented="fragmented")
closest_goal = recognizer.inference_phase(partial_sequence, maze_str_to_goal("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4"), 0.5)
print(f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: (4, 4)")
