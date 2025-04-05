from gr_libs.environment.environment import MINIGRID, QLEARNING
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.tabular.tabular_q_learner import TabularQLearner
from gr_libs.ml.utils.format import random_subset_with_order
from gr_libs import ExpertBasedGraml

def run_graml_minigrid_tutorial():
    recognizer = ExpertBasedGraml(
        domain_name=MINIGRID,
        env_name="MiniGrid-SimpleCrossingS13N4"
    )

    recognizer.domain_learning_phase(base_goals=[(11,1), (11,11), (1,11), (7,11), (8,1), (10,6), (6,9), (11,3), (11,5)],
                                    train_configs=[(QLEARNING, 100000) for _ in range(9)])

    recognizer.goals_adaptation_phase(
        dynamic_goals = [(11,1), (11,11), (1,11)],
        dynamic_train_configs=[(QLEARNING, 100000) for _ in range(3)] # for expert sequence generation.
    )
    # TD3 is different from recognizer and expert algorithms, which are SAC #
    actor = TabularQLearner(domain_name="minigrid", problem_name="MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0", algorithm=QLEARNING, num_timesteps=100000)
    actor.learn()
    # sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
    full_sequence = actor.generate_observation(
        action_selection_method=stochastic_amplified_selection,
        random_optimalism=True, # the noise that's added to the actions
    )

    partial_sequence = random_subset_with_order(full_sequence, (int)(0.5 * len(full_sequence)), is_consecutive=False)
    closest_goal = recognizer.inference_phase(partial_sequence, (11,1), 0.5)
    print(f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: (11, 1)")

if __name__ == "__main__":
    run_graml_minigrid_tutorial()