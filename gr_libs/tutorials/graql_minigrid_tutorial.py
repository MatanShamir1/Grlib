from gr_libs import Graql
from gr_libs.environment._utils.utils import domain_to_env_property
from gr_libs.environment.environment import MINIGRID, QLEARNING
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.tabular.tabular_q_learner import TabularQLearner
from gr_libs.ml.utils.format import random_subset_with_order


def run_graql_minigrid_tutorial():
    recognizer = Graql(domain_name="minigrid", env_name="MiniGrid-SimpleCrossingS13N4")

    # Graql doesn't have a domain learning phase, so we skip it

    recognizer.goals_adaptation_phase(
        dynamic_goals=[(11, 1), (11, 11), (1, 11)],
        dynamic_train_configs=[
            (QLEARNING, 100000) for _ in range(3)
        ],  # for expert sequence generation.
    )

    property_type = domain_to_env_property(MINIGRID)
    env_property = property_type("MiniGrid-SimpleCrossingS13N4")

    # TD3 is different from recognizer and expert algorithms, which are SAC #
    actor = TabularQLearner(
        domain_name="minigrid",
        problem_name="MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",
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
    closest_goal = recognizer.inference_phase(partial_sequence, (11, 1), 0.5)
    print(
        f"closest_goal returned by Graql: {closest_goal}\nactual goal actor aimed towards: (11, 1)"
    )
    return closest_goal, (11, 1)


if __name__ == "__main__":
    run_graql_minigrid_tutorial()
