import random
import sys
from typing import List
from consts import MAZE_PROBLEMS, MINIGRID_PROBLEMS
from ml.neural.model import NeuralAgent
from ml.utils.storage import set_global_storage_configs
from recognizer.graql_recognizer import GraqlRecognizer
import scripts.file_system as file_system
from recognizer import GramlRecognizer
from ml import TabularQLearner
from metrics.metrics import greedy_selection
import os
import pickle

def init(recognizer_str:str, is_fragmented:bool, collect_statistics:bool, is_inference_same_length_sequences:bool=None, is_learn_same_length_sequences:bool=None, task:str=None):
#     world = input("Welcome to GRAML!\n \
# As a proper ODGR framework should be, I'm interactive.\n \
# I will ask for a domain, then some initial goals, and then you could enter new goals or observations for recognition right after I tell you I'm ready.\n \
# Let's start with you specifying some things about the environment.\n\n \
# What's the world we're in?")
	if task == 'MINIGRID':
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
		def goal_to_task_str(tuply):
			tuply = tuply[1:-1] # remove the braces
			#print(tuply)
			nums = tuply.split(',')
			#print(nums)
			return f'MiniGrid-SimpleCrossingS13N4-DynamicGoal-{nums[0]}x{nums[1]}-v0'

		def problem_list_to_str_tuple(problems : List[str]):
			return '_'.join([f"[{s.split('-')[-2]}]" for s in problems])

	
	elif task == "MAZE":
		problem_name = "PointMaze-FourRoomsEnv-11x11-3-PROBLEMS"
		env_name, problem_list = MAZE_PROBLEMS[problem_name]
		learner_type = NeuralAgent
		dynamic_goals = ['(7,3)', '(3,7)', '(6,4)', '(4,6)', '(3,3)', '(6,6)']
		if recognizer_str == "graql":
			print("Can't support GR as RL recognition yet. ask ben to give his framework here to evaluate next to his.")
			exit(1)
		else:
			recognizer_type = GramlRecognizer
		def goal_to_task_str(tuply):
			tuply = tuply[1:-1] # remove the braces
			#print(tuply)
			nums = tuply.split(',')
			#print(nums)
			return f'PointMaze-FourRoomsEnvDense-11x11-Goal-{nums[0]}x{nums[1]}'

		def problem_list_to_str_tuple(problems : List[str]):
			return '_'.join([f"[{s.split('-')[-1]}]" for s in problems])

	else:
		print("I currently only support minigrid and maze. I promise it will change in the future!")
		exit(1)
  
	recognizer = recognizer_type(learner_type, env_name, problem_list, exploration_rates=[None, None], is_fragmented=is_fragmented, is_inference_same_length_sequences=is_inference_same_length_sequences, is_learn_same_length_sequences=is_learn_same_length_sequences, collect_statistics=collect_statistics, goal_to_task_str=goal_to_task_str)
	# print("### STARTING DOMAIN LEARNING PHASE ###")
	recognizer.domain_learning_phase(problem_list_to_str_tuple=problem_list_to_str_tuple)
	recognizer.goals_adaptation_phase(dynamic_goals)
 
	# experiments - feel free to change according to task...
	task_num, correct = 0, 0
	for goal in dynamic_goals:
		for percentage in [0.3, 0.5, 0.7, 0.9, 1]:
			agent = learner_type(env_name=env_name, problem_name=goal_to_task_str(goal))
			agent.learn()
			sequence = agent.generate_partial_observation(action_selection_method=greedy_selection, percentage=percentage, is_fragmented=is_fragmented, save_fig=True, random_optimalism=True)
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
	assert sys.argv[6] in ["MAZE", "MINIGRID"]
	if sys.argv[1] == "graml":
		set_global_storage_configs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
		init(recognizer_str=sys.argv[1], is_fragmented=sys.argv[2] == "fragmented_partial_obs", is_inference_same_length_sequences=sys.argv[3] == "inference_same_length", is_learn_same_length_sequences=sys.argv[4] == "learn_same_length", collect_statistics=sys.argv[5] == "collect_statistics", task=sys.argv[6])
	else: # graql
		set_global_storage_configs(sys.argv[1], sys.argv[2])
		init(recognizer_str=sys.argv[1], is_fragmented=sys.argv[2] == "fragmented_partial_obs", collect_statistics=sys.argv[3] == "collect_statistics", task=sys.argv[6])

	