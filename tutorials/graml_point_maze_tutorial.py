
from stable_baselines3 import SAC, TD3
from gr_libs.environment.environment import POINT_MAZE, PointMazeProperty
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent
from gr_libs.ml.utils.format import random_subset_with_order
from gr_libs.recognizer.graml.graml_recognizer import ExpertBasedGraml

def run_graml_point_maze_tutorial():
    recognizer = ExpertBasedGraml(
        domain_name=POINT_MAZE,
        env_name="PointMaze-FourRoomsEnvDense-11x11"
    )

    recognizer.domain_learning_phase(
        [(9,1), (9,9), (1,9), (3,3), (3,4), (8,2), (3,7), (2,8)],
        [(SAC, 200000) for _ in range(8)]
    )

    recognizer.goals_adaptation_phase(
        dynamic_goals = [(4,4), (7,3), (3,7)],
        dynamic_train_configs=[(SAC, 200000) for _ in range(3)] # for expert sequence generation.
    )

    # TD3 is different from recognizer and expert algorithms, which are SAC #
    actor = DeepRLAgent(domain_name="point_maze", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-4x4", algorithm=TD3, num_timesteps=200000)
    actor.learn()
    # sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
    full_sequence = actor.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True, # the noise that's added to the actions
    )

    partial_sequence = random_subset_with_order(full_sequence, (int)(0.5 * len(full_sequence)))
    closest_goal = recognizer.inference_phase(partial_sequence, PointMazeProperty("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4").str_to_goal(), 0.5)
    print(f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: (4, 4)")

if __name__ == "__main__":
    run_graml_point_maze_tutorial()