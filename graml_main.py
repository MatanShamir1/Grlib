import random
from consts import MINIGRID_PROBLEMS
import scripts.file_system as file_system
from recognizer import GramlRecognizer, goal_to_minigrid_str
from ml import TabularQLearner
from metrics.metrics import greedy_selection
import os
import pickle

def init():
#     world = input("Welcome to GRAML!\n \
# As a proper ODGR framework should be, I'm interactive.\n \
# I will ask for a domain, then some initial goals, and then you could enter new goals or observations for recognition right after I tell you I'm ready.\n \
# Let's start with you specifying some things about the environment.\n\n \
# What's the world we're in?")
	world = 'MINIGRID'
	if world == 'MINIGRID':
		# problem_name  = input("Please specify the problem name from consts.py. This is my domain theory and an initial set of prototype goals I picked manually, constituting my domain learning time. for example: MiniGrid-Simple-9x9-3-PROBLEMS")
		problem_name = 'MiniGrid-Walls-13x13-10-PROBLEMS'
		# grid_size = input("Please specify the grid size. for example: 9")
		grid_size = 13
		env_name, problem_list = MINIGRID_PROBLEMS[problem_name]
		observation_path = file_system.get_observations_path(env_name=env_name)
		observations_paths = file_system.get_observations_paths(path=observation_path)
		
		# get a partial observability of the actor. the goal to which it goes is the first goal from the list, hence the [0].
		# GET THIS OUT TO A FUNCTION
		actor = TabularQLearner(env_name=env_name, problem_name = problem_list[0])
		actor.learn()
		steps = actor.generate_observation(greedy_selection)
		if not os.path.exists(observation_path):
			os.makedirs(observation_path)
		with open(f'{observation_path}/obs1.0.pkl', 'wb') as f:
			pickle.dump(steps, f)
		file_system.create_partial_observabilities_files(env_name=env_name, observabilities=[0.1, 0.3, 0.5, 0.7])
		file_system.print_md5(file_path_list=observations_paths)
		
		recognizer = GramlRecognizer(TabularQLearner, env_name, problem_list, grid_size, is_continuous=False)
		print("### STARTING DOMAIN LEARNING PHASE ###")
		recognizer.domain_learning_phase()
		recognizer.goals_adaptation_phase(['(6,1)', '(11,3)', '(11,5)', '(11,8)', '(1,7)', '(5,9)'])
		task_num, correct = 0, 0
		for goal in ['(6,1)', '(11,3)', '(11,5)', '(11,8)', '(1,7)', '(5,9)']:
			for percentage in [0.3, 0.5, 0.7, 0.9, 1]:
				agent = TabularQLearner(env_name=env_name, problem_name=goal_to_minigrid_str(goal))
				agent.learn()
				sequence = agent.generate_partial_observation(action_selection_method=greedy_selection, percentage=percentage)
				closest_goal = recognizer.inference_phase(sequence, task_num)
				print(f'real goal {goal}, closest goal is: {closest_goal}')
				if all(a == b for a, b in zip(goal, closest_goal)):
					correct += 1
				task_num += 1

		print(f'correct: {correct}\n total tasks: {task_num}\n accuracy: {correct/task_num}')
    
		

	else:
		print("I currently only support minigrid. I promise it will change in the future!")
		exit(1)

# def interactive_recognition(observation_path, observations_paths, recognizer):
#     initial_goal_set = input("Please specify an initial set of goals for me. I will perform goals adaptation time now.")
#     pass

def main():
	init()


if __name__ == "__main__":
	main()
	