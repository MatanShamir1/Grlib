from gr_libs import ExpertBasedGraml
from gr_libs.environment._utils.utils import domain_to_env_property
from gr_libs.environment.environment import MINIGRID, QLEARNING
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.tabular.tabular_q_learner import TabularQLearner
from gr_libs.ml.utils.format import random_subset_with_order


def run_ExpertBasedGraml_LavaMinigrid_tutorial():
    recognizer = ExpertBasedGraml(
        domain_name=MINIGRID, env_name="MiniGrid-LavaCrossingS9N2"
    )

    recognizer.domain_learning_phase(
        {
            "bg": {
                "goals": [(1, 3), (6, 5), (4, 7)],
                "train_configs": [(QLEARNING, 100000) for _ in range(3)],
            }
        }
    )

    recognizer.goals_adaptation_phase(
        dynamic_goals=[(1, 3), (6, 5), (4, 7), (2, 5)],
        dynamic_train_configs=[
            (QLEARNING, 100000) for _ in range(4)
        ],  # for expert sequence generation.
    )

    property_type = domain_to_env_property(MINIGRID)
    env_property = property_type("MiniGrid-LavaCrossingS9N2")

    # TD3 is different from recognizer and expert algorithms, which are SAC #
    actor = TabularQLearner(
        domain_name="minigrid",
        problem_name="MiniGrid-LavaCrossingS9N2-DynamicGoal-4x7-v0",
        env_prop=env_property,
        algorithm=QLEARNING,
        num_timesteps=100000,
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
    closest_goal = recognizer.inference_phase(partial_sequence, (4, 7), 0.5)
    print(
        f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: (4, 7)"
    )


if __name__ == "__main__":
    run_ExpertBasedGraml_LavaMinigrid_tutorial()
