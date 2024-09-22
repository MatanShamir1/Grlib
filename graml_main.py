import random
import sys
from typing import List

from stable_baselines3 import SAC, TD3
from consts import MAZE_PROBLEMS, MINIGRID_PROBLEMS, PARKING_PROBLEMS
from ml.neural.SB3model import NeuralAgent
from ml.neural.ppo import PPOAlgo
from ml.utils.format import minigrid_str_to_goal, maze_str_to_goal
from ml.utils.storage import set_global_storage_configs
from recognizer.graml_recognizer import MCTS_BASED, AGENT_BASED, GC_AGENT_BASED
from recognizer.graql_recognizer import GraqlRecognizer
from recognizer import GramlRecognizer
from ml import TabularQLearner
from metrics.metrics import greedy_selection, measure_sequence_similarity

def init(recognizer_str:str, is_fragmented:bool, collect_statistics:bool, is_inference_same_length_sequences:bool=None, is_learn_same_length_sequences:bool=None, task:str=None):
#     world = input("Welcome to GRAML!\n \
# As a proper ODGR framework should be, I'm interactive.\n \
# I will ask for a domain, then some initial goals, and then you could enter new goals or observations for recognition right after I tell you I'm ready.\n \
# Let's start with you specifying some things about the environment.\n\n \
# What's the world we're in?")
	domain, task = task.split(":")
	if domain == 'MINIGRID':
		gc_goal_set = None
		# problem_name  = input("Please specify the problem name from consts.py. This is my domain theory and an initial set of prototype goals I picked manually, constituting my domain learning time. for example: MiniGrid-Simple-9x9-3-PROBLEMS")
		problem_name = 'MiniGrid-Walls-13x13-10-PROBLEMS'
		# grid_size = input("Please specify the grid size. for example: 9")
		env_name, problem_list = MINIGRID_PROBLEMS[problem_name]
		learner_type = TabularQLearner
		if recognizer_str == "graql":
			recognizer_type = GraqlRecognizer
		else:
			recognizer_type = GramlRecognizer
   
		dynamic_goals = ['(6,1)', '(11,3)', '(11,5)', '(11,8)', '(1,7)', '(5,9)']
		train_configs = [(None, None), (None, None), (None, None), (None, None), (None, None)] # irrelevant for now, check how to get rid of
		specified_rl_algorithm = None
		specified_rl_algorithm_inference = None
		goals_adaptation_sequence_generation_method = MCTS_BASED
		task_str_to_goal = minigrid_str_to_goal
		input_size = 4; hidden_size = 8; batch_size = 16
		def goal_to_task_str(tuply):
			tuply = tuply[1:-1] # remove the braces
			#print(tuply)
			nums = tuply.split(',')
			#print(nums)
			return f'MiniGrid-SimpleCrossingS13N4-DynamicGoal-{nums[0]}x{nums[1]}-v0'

		def problem_list_to_str_tuple(problems : List[str]):
			return '_'.join([f"[{s.split('-')[-2]}]" for s in problems])


	elif domain == "MAZE":
		gc_goal_set = None
		if task == "FOUR_ROOMS":
			problem_name = "PointMaze-FourRoomsEnv-11x11-3-PROBLEMS"
			dynamic_goals = ['(7,3)', '(3,7)', '(6,4)', '(4,4)', '(3,4)']
			train_configs = [(None, 200000), (None, 200000), (None, 200000)]
			env_str = "FourRooms"
		else: # task == "OBSTACLES"
			problem_name = "PointMaze-ObstaclesEnv-11x11-3-PROBLEMS"
			dynamic_goals = ['(5,5)', '(3,6)', '(6,3)']
			train_configs = [(None, 200000), (None, 200000), (None, 200000)]
			env_str = "Obstacles"
		env_name, problem_list = MAZE_PROBLEMS[problem_name]
		goals_adaptation_sequence_generation_method = AGENT_BASED
		task_str_to_goal = maze_str_to_goal
		input_size = 6; hidden_size = 8; batch_size = 32
		if recognizer_str == "graql":
			learner_type = PPOAlgo
			recognizer_type = GraqlRecognizer
			specified_rl_algorithm = None # bens algorithm doesn't accept a specific algorithm... only ppo
			specified_rl_algorithm_inference = TD3 # inference is not with bens env anyway.
		else:
			learner_type = NeuralAgent
			recognizer_type = GramlRecognizer
			specified_rl_algorithm = SAC
			specified_rl_algorithm_inference = TD3
		def goal_to_task_str(tuply):
			tuply = tuply[1:-1] # remove the braces
			#print(tuply)
			nums = tuply.split(',')
			#print(nums)
			return f'PointMaze-{env_str}EnvDense-11x11-Goal-{nums[0]}x{nums[1]}'

		def problem_list_to_str_tuple(problems : List[str]):
			return '_'.join([f"[{s.split('-')[-1]}]" for s in problems])

	elif domain == "PARKING":
		if task == "AGENT":
			train_configs = [(None, 200000), (None, 200000), (None, 200000)]
			problem_name = "ParkingEnvContinuous-Hard-4-Problems"
			gc_goal_set = None
			goals_adaptation_sequence_generation_method = AGENT_BASED
			def goal_to_task_str(goal_index):
				return f'Parking-S-10-PC--GI-{goal_index}-v0'
		else: # task == "GC_AGENT"
			train_configs = [(None, 300000)]
			goals_adaptation_sequence_generation_method = GC_AGENT_BASED
			problem_name = "ParkingEnvUniversal-Hard-4-Problems"
			gc_goal_set = [i for i in range(1,9)]
			def goal_to_task_str(goal_index):
				return f'parking-v0'
		env_name, problem_list = PARKING_PROBLEMS[problem_name]
		dynamic_goals = ['3', '6', '7']
		task_str_to_goal = maze_str_to_goal
		input_size = 8; hidden_size = 8; batch_size = 32
		if recognizer_str == "graql":
			learner_type = PPOAlgo
			recognizer_type = GraqlRecognizer
			specified_rl_algorithm = None # bens algorithm doesn't accept a specific algorithm... only ppo
			specified_rl_algorithm_inference = TD3 # inference is not with bens env anyway.
		else:
			learner_type = NeuralAgent
			recognizer_type = GramlRecognizer
			specified_rl_algorithm = SAC
			specified_rl_algorithm_inference = TD3
		

		def problem_list_to_str_tuple(problems : List[str]):
			return '_'.join([f"[{s.split('-')[-2]}]" for s in problems])

	else:
		print("I currently only support minigrid and maze. I promise it will change in the future!")
		exit(1)
  
	recognizer = recognizer_type(learner_type, env_name, problem_list, train_configs=train_configs, task_str_to_goal=task_str_to_goal,
								is_fragmented=is_fragmented, is_inference_same_length_sequences=is_inference_same_length_sequences,
								is_learn_same_length_sequences=is_learn_same_length_sequences, collect_statistics=collect_statistics,
								goal_to_task_str=goal_to_task_str, specified_rl_algorithm=specified_rl_algorithm,
								goals_adaptation_sequence_generation_method=goals_adaptation_sequence_generation_method,
								gc_sequence_generation = (goals_adaptation_sequence_generation_method==GC_AGENT_BASED),
        						gc_goal_set=gc_goal_set)

	# print("### STARTING DOMAIN LEARNING PHASE ###")
	recognizer.domain_learning_phase(problem_list_to_str_tuple=problem_list_to_str_tuple, input_size=input_size, hidden_size=hidden_size, batch_size=batch_size)
	recognizer.goals_adaptation_phase(dynamic_goals)
 
	# experiments
	task_num, correct = 0, 0
	for goal in dynamic_goals:
		for percentage in [0.3, 0.5, 0.7, 0.9, 1]:
			kwargs = {"env_name":env_name, "problem_name":goal_to_task_str(goal)}
			if specified_rl_algorithm_inference: kwargs["algorithm"] = specified_rl_algorithm_inference
			if train_configs[0][0]: kwargs["exploration_rate"] = train_configs[0][0]
			if train_configs[0][1]: kwargs["num_timesteps"] = train_configs[0][1]
			agent = learner_type(**kwargs)
			agent.learn()
			sequence = agent.generate_partial_observation(action_selection_method=greedy_selection, percentage=percentage, is_fragmented=is_fragmented, save_fig=True, random_optimalism=True)
			# need to measure the similarity of the full plan before it was 'random_with_subset" and all that... TODO
			sequence_similarities = [measure_sequence_similarity(sequence, aps) for aps in [recognizer.get_goal_plan(g) for g in dynamic_goals]]
			closest_goal = recognizer.inference_phase(sequence, goal, percentage)
			# print(f'real goal {goal}, closest goal is: {closest_goal}')
			if all(a == b for a, b in zip(goal, closest_goal)):
				correct += 1
			task_num += 1

	print(f'correct: {correct}\n total tasks: {task_num}\n accuracy: {correct/task_num}')
# def interactive_recognition(observation_path, observations_paths, recognizer):
#     initial_goal_set = input("Please specify an initial set of goals for me. I will perform goals adaptation time now.")
#     pass


if __name__ == "__main__":
	assert len(sys.argv) == 7
	assert sys.argv[1] in ["graml"] and sys.argv[2] in ["continuing_partial_obs", "fragmented_partial_obs"] and sys.argv[3] in ["inference_same_length", "inference_diff_length"] and sys.argv[4] in ["learn_same_length", "learn_diff_length"] and sys.argv[5] in ['no_collect_statistics', 'collect_statistics'] \
			or len(sys.argv) == 4 and sys.argv[1] in ["graql"] and sys.argv[2] in ["continuing_partial_obs", "fragmented_partial_obs"] and sys.argv[3] in ['no_collect_statistics', 'collect_statistics'] \
	   ,f"Assertion failed: incorrect arguments.\nExample 1: \n\t python graml_main.py graml [continuing_partial_obs/fragmented_partial_obs] [inference_same_length/inference_diff_length] [learn_same_length/learn_diff_length] [collect_statistics/no_collect_statistics]\nExample 2: \n\t python graml_main.py graql [continuing_partial_obs/fragmented_partial_obs] [collect_statistics/no_collect_statistics]"
	assert sys.argv[6] in ["MAZE:FOUR_ROOMS", "MAZE:OBSTACLES", "MINIGRID", "PARKING:GC_AGENT", "PARKING:AGENT"]
	if sys.argv[1] == "graml":
		set_global_storage_configs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
		init(recognizer_str=sys.argv[1], is_fragmented=sys.argv[2] == "fragmented_partial_obs", is_inference_same_length_sequences=sys.argv[3] == "inference_same_length", is_learn_same_length_sequences=sys.argv[4] == "learn_same_length", collect_statistics=sys.argv[5] == "collect_statistics", task=sys.argv[6])
	else: # graql
		set_global_storage_configs(sys.argv[1], sys.argv[2])
		init(recognizer_str=sys.argv[1], is_fragmented=sys.argv[2] == "fragmented_partial_obs", collect_statistics=sys.argv[3] == "collect_statistics", task=sys.argv[6])

	
# python graml_main.py graml fragmented_partial_obs inference_same_length learn_diff_length collect_statistics MAZE