
import numpy as np
from stable_baselines3 import PPO, SAC
from grlib.environment.environment import PANDA, GCEnvProperty, PandaProperty
from grlib.environment.utils.utils import domain_to_env_property
from grlib.metrics.metrics import stochastic_amplified_selection
from grlib.ml.neural.deep_rl_learner import DeepRLAgent, GCDeepRLAgent
from grlib.ml.utils.format import random_subset_with_order
from grlib.recognizer.graml.graml_recognizer import GCGraml

# Consider extracting all these to "default point_maze (or every other domain) variables" module which would simplify things like the problem_list_to_str_tuple function, sizes of inputs, etc.
recognizer = GCGraml( # TODO make these tutorials into pytests
    domain_name=PANDA,
	env_name="PandaMyReachDense"
)
recognizer.domain_learning_phase(base_goals=[np.array([PandaProperty.sample_goal()]) for _ in range(1,30)],
					             train_configs=[(SAC, 800000)])
recognizer.goals_adaptation_phase(dynamic_goals=[np.array([[-0.1, -0.1, 0.1]]), np.array([[-0.1, 0.1, 0.1]]), np.array([[0.2, 0.2, 0.1]])])
# TD3 is different from recognizer and expert algorithms, which are SAC #
property_type = domain_to_env_property(PANDA)
env_property = property_type("PandaMyReachDense")
problem_name = env_property.goal_to_problem_str(np.array([[-0.1, -0.1, 0.1]]))
actor = DeepRLAgent(domain_name=PANDA, problem_name=problem_name, algorithm=PPO, num_timesteps=400000)
actor.learn()
# sample is generated stochastically to simulate suboptimal behavior, noise is added to the actions values #
full_sequence = actor.generate_observation(
    action_selection_method=stochastic_amplified_selection,
    random_optimalism=True, # the noise that's added to the actions
)

partial_sequence = random_subset_with_order(full_sequence, (int)(0.5 * len(full_sequence)), is_consecutive=False)
closest_goal = recognizer.inference_phase(partial_sequence, np.array([[-0.1, -0.1, 0.1]]), 0.5)
print(f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: [-0.1, -0.1, 0.1]")
