from stable_baselines3 import SAC, TD3

from gr_libs.environment._utils.utils import domain_to_env_property
from gr_libs.environment.environment import POINT_MAZE, PointMazeProperty
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent
from gr_libs.ml.utils.format import random_subset_with_order
from gr_libs.recognizer.graml.graml_recognizer import ExpertBasedGraml


def run_ExpertBasedGraml_PointMazeObstacles_tutorial():
    recognizer = ExpertBasedGraml(
        domain_name=POINT_MAZE, env_name="PointMaze-ObstaclesEnvDense-11x11"
    )

    recognizer.domain_learning_phase(
        {
            "bg": {
                "goals": [
                    (5, 1),
                    (9, 9),
                    (1, 5),
                    (6, 4),
                    (4, 6),
                    (6, 6),
                    (7, 7),
                ],
                "train_configs": [(SAC, 400000) for _ in range(7)],
            }
        }
    )

    recognizer.goals_adaptation_phase(
        dynamic_goals=[(5, 5), (7, 4), (4, 7)],
        dynamic_train_configs=[
            (SAC, 200000) for _ in range(3)
        ],  # for expert sequence generation.
    )

    property_type = domain_to_env_property(POINT_MAZE)
    env_property = property_type("PointMaze-ObstaclesEnvDense-11x11")

    # TD3 is different from recognizer and expert algorithms, which are SAC #
    actor = DeepRLAgent(
        domain_name="point_maze",
        problem_name="PointMaze-ObstaclesEnvDense-11x11-Goal-7x4",
        env_prop=env_property,
        algorithm=TD3,
        num_timesteps=200000,
    )
    actor.learn()
    # sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
    full_sequence = actor.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True,  # the noise that's added to the actions
    )

    partial_sequence = random_subset_with_order(
        full_sequence, (int)(0.5 * len(full_sequence))
    )
    closest_goal = recognizer.inference_phase(
        partial_sequence,
        PointMazeProperty("PointMaze-ObstaclesEnvDense-11x11-Goal-7x4").str_to_goal(),
        0.5,
    )
    print(
        f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: (7, 4)"
    )


if __name__ == "__main__":
    run_ExpertBasedGraml_PointMazeObstacles_tutorial()
