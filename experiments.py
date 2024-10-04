import argparse
import random
import sys
from typing import List

from metrics import metrics
from recognizer.graml.graml_recognizer import MCTS_BASED, AGENT_BASED, GC_AGENT_BASED
from recognizer.graql.graql_recognizer import GraqlRecognizer
from recognizer import GramlRecognizer
from metrics.metrics import greedy_selection
from ml.utils.format import random_subset_with_order
from recognizer.utils import recognizer_str_to_obj
from ml.utils.storage import set_global_storage_configs

# keep this import last, dependent on lots of initialized modules.
from consts import PROBLEMS

def run_experiments(args):
	# recognizer_kwargs = {}
	# if domain == 'MINIGRID':
	# 	# problem_name  = input("Please specify the problem name from consts.py. This is my domain theory and an initial set of prototype goals I picked manually, constituting my domain learning time. for example: MiniGrid-Simple-9x9-3-PROBLEMS")
	# 	problem_name = 'MiniGrid-Walls-13x13-10-PROBLEMS'
	# 	# grid_size = input("Please specify the grid size. for example: 9")
	# 	env_name, problem_list = MINIGRID_PROBLEMS[problem_name]
	# 	learner_type = TabularQLearner
	# 	dynamic_goals = ['(6,1)', '(11,3)', '(11,5)', '(11,8)', '(1,7)', '(5,9)']
	# 	recognizer_kwargs["train_configs"] = [(None, None), (None, None), (None, None), (None, None), (None, None)] # irrelevant for now, check how to get rid of
	# 	recognizer_kwargs["task_str_to_goal"] = minigrid_str_to_goal
	# 	input_size = 4; hidden_size = 8; batch_size = 16
	# 	recognizer_kwargs["problems"] = problem_list
	# 	def goal_to_task_str(tuply):
	# 		tuply = tuply[1:-1] # remove the braces
	# 		#print(tuply)
	# 		nums = tuply.split(',')
	# 		#print(nums)
	# 		return f'MiniGrid-SimpleCrossingS13N4-DynamicGoal-{nums[0]}x{nums[1]}-v0'

	# 	if recognizer_str == "graql":
	# 		recognizer_kwargs["evaluation_function"] = metrics.kl_divergence_norm_softmax
	# 		recognizer_type = GraqlRecognizer
	# 	else:
	# 		recognizer_type = GramlRecognizer
	# 		recognizer_kwargs["is_fragmented"] = is_fragmented
	# 		recognizer_kwargs["goals_adaptation_sequence_generation_method"] = MCTS_BASED
	# 		recognizer_kwargs["is_inference_same_length_sequences"] = is_inference_same_length_sequences
	# 		recognizer_kwargs["is_learn_same_length_sequences"] = is_learn_same_length_sequences

	# 	recognizer_kwargs["goal_to_task_str"] = goal_to_task_str
	# 	goal_to_task_str_inference = goal_to_task_str

	# 	def problem_list_to_str_tuple(problems : List[str]):
	# 		return '_'.join([f"[{s.split('-')[-2]}]" for s in problems])


	# elif domain == "MAZE":
	# 	if task == "FOUR_ROOMS":
	# 		problem_name = "PointMaze-FourRoomsEnv-11x11-3-PROBLEMS"
	# 		dynamic_goals = ['(7,3)', '(3,7)', '(4,4)', '(3,4)']
	# 		recognizer_kwargs["train_configs"] = [(None, 200000), (None, 200000), (None, 200000)]
	# 		env_str = "FourRooms"
	# 	else: # task == "OBSTACLES"
	# 		# problem_name = "PointMaze-ObstaclesEnvEasy-11x11-3-PROBLEMS"
	# 		problem_name = "PointMaze-ObstaclesEnv-11x11-3-PROBLEMS"
	# 		dynamic_goals = ['(5,5)', '(3,6)', '(6,3)', '(7,4)', '(4,7)']
	# 		# train_configs = [(None, 200000), (None, 200000), (None, 200000), (None, 200000), (None, 200000)]
	# 		recognizer_kwargs["train_configs"] = [(None, 200000), (None, 200000), (None, 200000)]
	# 		env_str = "Obstacles"
	# 	env_name, problem_list = MAZE_PROBLEMS[problem_name]
	# 	recognizer_kwargs["task_str_to_goal"] = maze_str_to_goal
	# 	input_size = 6; hidden_size = 8; batch_size = 32
	# 	recognizer_kwargs["specified_rl_algorithm"] = SAC
	# 	specified_rl_algorithm_inference = TD3
	# 	recognizer_kwargs["problems"] = problem_list

	# 	def goal_to_task_str(tuply):
	# 		tuply = tuply[1:-1] # remove the braces
	# 		#print(tuply)
	# 		nums = tuply.split(',')
	# 		#print(nums)
	# 		return f'PointMaze-{env_str}EnvDense-11x11-Goal-{nums[0]}x{nums[1]}'

	# 	if recognizer_str == "graql":
	# 		recognizer_kwargs["evaluation_function"] = mean_wasserstein_distance
	# 		recognizer_type = GraqlRecognizer
			
	# 		recognizer_kwargs["train_configs"] = [(None, 200000), (None, 200000), (None, 200000), (None, 200000), (None, 200000)] # update to 5 dynamic goals training configs
	# 	else:
	# 		recognizer_type = GramlRecognizer
	# 		recognizer_kwargs["is_fragmented"] = is_fragmented
	# 		recognizer_kwargs["goals_adaptation_sequence_generation_method"] = AGENT_BASED
	# 		recognizer_kwargs["is_inference_same_length_sequences"] = is_inference_same_length_sequences
	# 		recognizer_kwargs["is_learn_same_length_sequences"] = is_learn_same_length_sequences

	# 	recognizer_kwargs["goal_to_task_str"] = goal_to_task_str
	# 	goal_to_task_str_inference = goal_to_task_str
	# 	learner_type = NeuralAgent

	# 	def problem_list_to_str_tuple(problems : List[str]):
	# 		return '_'.join([f"[{s.split('-')[-1]}]" for s in problems])

	# elif domain == "PARKING":
	# 	if task == "AGENT":
	# 		recognizer_kwargs["train_configs"] = [(None, 300000), (None, 300000), (None, 300000), (None, 300000), (None, 300000)]
	# 		problem_name = "ParkingEnvContinuous-Hard-4-Problems"
	# 		recognizer_kwargs["goals_adaptation_sequence_generation_method"] = AGENT_BASED
	# 		def goal_to_task_str(goal_index):
	# 			return f'Parking-S-14-PC--GI-{goal_index}-v0'

	# 		recognizer_kwargs["goal_to_task_str"] = goal_to_task_str
	# 		goal_to_task_str_inference = goal_to_task_str
	# 	else: # task == "GC_AGENT"
	# 		recognizer_kwargs["train_configs"] = [(None, 300000)]
	# 		recognizer_kwargs["goals_adaptation_sequence_generation_method"] = GC_AGENT_BASED
	# 		problem_name = "ParkingEnvUniversal-Hard-4-Problems"
	# 		recognizer_kwargs["gc_goal_set"] = [i for i in range(1,9)]
	# 		def goal_to_task_str(goal_index):
	# 			return f'parking-v0'
	# 		def goal_to_task_str_inference(goal_index):
	# 			return f'Parking-S-14-PC--GI-{goal_index}-v0'

	# 		recognizer_kwargs["goal_to_task_str"] = goal_to_task_str
	# 		goal_to_task_str_inference = goal_to_task_str_inference

	# 	env_name, problem_list = PARKING_PROBLEMS[problem_name]
	# 	# dynamic_goals = ['3', '6', '7']
	# 	dynamic_goals = ['1', '7', '14']
	# 	recognizer_kwargs["task_str_to_goal"] = parking_str_to_goal
	# 	input_size = 8; hidden_size = 8; batch_size = 32
	# 	learner_type = NeuralAgent
	# 	recognizer_kwargs["problems"] = problem_list
	# 	if recognizer_str == "graql":
	# 		recognizer_kwargs.pop("goals_adaptation_sequence_generation_method")
	# 		recognizer_kwargs["evaluation_function"] = mean_wasserstein_distance
	# 		recognizer_type = GraqlRecognizer
	# 	else:
	# 		recognizer_type = GramlRecognizer
	# 		recognizer_kwargs["is_fragmented"] = is_fragmented
	# 		recognizer_kwargs["is_inference_same_length_sequences"] = is_inference_same_length_sequences
	# 		recognizer_kwargs["is_learn_same_length_sequences"] = is_learn_same_length_sequences

	# 	recognizer_kwargs["specified_rl_algorithm"] = SAC
	# 	specified_rl_algorithm_inference = TD3
		
	# 	def problem_list_to_str_tuple(problems : List[str]):
	# 		return '_'.join([f"[{s.split('-')[-2]}]" for s in problems])

	# elif domain == "KITCHEN":
	# 	if task == "COMB1":
	# 		recognizer_kwargs["train_configs"] = [(None, 200000), (None, 200000), (None, 200000), (None, 200000)]
	# 		problem_name = "FrankaKitchen-4-Problems"
	# 		def goal_to_task_str(goal_index):
	# 			return "FrankaKitchen-v1"
	# 		recognizer_kwargs["goal_to_task_str"] = goal_to_task_str
	# 		goal_to_task_str_inference = goal_to_task_str
	# 		recognizer_kwargs["tasks_to_complete"] = True
	# 	else: # task == "COMB2"
	# 		exit(1)
	# 		recognizer_kwargs["train_configs"] = [(None, 200000), (None, 200000), (None, 200000), (None, 200000)]
	# 		problem_name = "FrankaKitchen-4-Problems"
	# 		recognizer_kwargs["goals_adaptation_sequence_generation_method"] = AGENT_BASED
	# 		def goal_to_task_str(goal_index):
	# 			return f'Parking-S-10-PC--GI-{goal_index}-v0'
	# 		recognizer_kwargs["goal_to_task_str"] = goal_to_task_str
	# 		goal_to_task_str_inference = goal_to_task_str
	# 	env_name, problem_list = KITCHEN_PROBLEMS[problem_name]
	# 	dynamic_goals = ["microwave", "top burner", "hinge cabinet"]
	# 	recognizer_kwargs["task_str_to_goal"] = maze_str_to_goal
	# 	input_size = 8; hidden_size = 8; batch_size = 32
	# 	recognizer_kwargs["problems"] = problem_list
	# 	if recognizer_str == "graql":
	# 		recognizer_kwargs["evaluation_function"] = mean_wasserstein_distance
	# 		recognizer_type = GraqlRecognizer
	# 	else:
	# 		recognizer_type = GramlRecognizer
	# 		recognizer_kwargs["goals_adaptation_sequence_generation_method"] = AGENT_BASED
	# 		recognizer_kwargs["is_fragmented"] = is_fragmented
	# 		recognizer_kwargs["is_inference_same_length_sequences"] = is_inference_same_length_sequences
	# 		recognizer_kwargs["is_learn_same_length_sequences"] = is_learn_same_length_sequences

	# 	learner_type = NeuralAgent
	# 	recognizer_kwargs["specified_rl_algorithm"] = SAC
	# 	specified_rl_algorithm_inference = TD3


	# 	def problem_list_to_str_tuple(problems : List[str]):
	# 		return '_'.join([s for s in problems])


	# else:
	# 	print("I currently only support minigrid, maze, franka-kitchen and parking. I promise it will change in the future!")
	# 	exit(1)

	# kwargs for every recognizer:
	recognizer_type = recognizer_str_to_obj(args.recognizer)
	recognizer_kwargs = {}

	# relevant also for inference phase experiments
	domain_inputs = PROBLEMS[args.domain]
	env_name, = [x for x in [args.minigrid_env, args.parking_env, args.point_maze_env, args.franka_env] if isinstance(x, str)]
	task_inputs = domain_inputs["tasks"][env_name]
	task_str_to_goal = task_inputs[args.task]["task_str_to_goal"]
	learner_type = domain_inputs["additional_combined_recognizer_kwargs"]["learner_type"]
	dynamic_goals_problems, dynamic_train_configs = zip(*task_inputs[args.task]["dynamic_goals_train_configs"])
	specified_rl_algorithm_inference = None
	if "specified_rl_algorithm_inference" in domain_inputs["additional_combined_recognizer_kwargs"].keys():
		specified_rl_algorithm_inference = domain_inputs["additional_combined_recognizer_kwargs"]["specified_rl_algorithm_inference"]

	# relevant for all recognizers' inputs
	recognizer_kwargs["env_name"] = args.domain
	base_problems, base_train_configs = zip(*task_inputs[args.task]["base_goals_problems_train_configs"])
	recognizer_kwargs["problems"] = base_problems
	recognizer_kwargs["task_str_to_goal"] = task_str_to_goal
	recognizer_kwargs["method"] = learner_type
	specified_rl_algorithm = None
	if "specified_rl_algorithm" in domain_inputs["additional_combined_recognizer_kwargs"].keys():
		recognizer_kwargs["specified_rl_algorithm"] = domain_inputs["additional_combined_recognizer_kwargs"]["specified_rl_algorithm"]
	recognizer_kwargs["collect_statistics"] = args.collect_stats
	if "tasks_to_complete" in domain_inputs["additional_combined_recognizer_kwargs"].keys(): recognizer_kwargs["tasks_to_complete"] = True
	recognizer_kwargs["train_configs"] = base_train_configs

	# each recognizer's specific inputs
	if args.recognizer == "graml":
		recognizer_kwargs["is_inference_same_length_sequences"] = args.inference_same_seq_len
		recognizer_kwargs["is_learn_same_length_sequences"] = args.learn_same_seq_len
		recognizer_kwargs["partial_obs_type"] = args.partial_obs_type
	recognizer_kwargs.update(task_inputs[args.task][f"additional_{args.recognizer}_kwargs"])
	
	# initialize recognizer
	recognizer = recognizer_type(**recognizer_kwargs)
	recognizer.domain_learning_phase()
	recognizer.goals_adaptation_phase(dynamic_goals_problems=dynamic_goals_problems, dynamic_train_configs=dynamic_train_configs)
 
	# experiments
	task_num, correct = 0, 0
	for problem in dynamic_goals_problems:
		goal = str(task_str_to_goal(problem))
		for percentage in [0.3, 0.5, 0.7, 0.9, 1]:
			kwargs = {"env_name":env_name, "problem_name":problem}
			if specified_rl_algorithm_inference: kwargs["algorithm"] = specified_rl_algorithm_inference
			if recognizer_kwargs["train_configs"][0][0]: kwargs["exploration_rate"] = recognizer_kwargs["train_configs"][0][0]
			if recognizer_kwargs["train_configs"][0][1]: kwargs["num_timesteps"] = recognizer_kwargs["train_configs"][0][1]
			agent = learner_type(**kwargs)
			agent.learn()
			# sequence = agent.generate_partial_observation(action_selection_method=greedy_selection, percentage=percentage, is_fragmented=is_fragmented, save_fig=True, random_optimalism=True)
			if recognizer_type == GraqlRecognizer:
				sequence = agent.generate_observation(action_selection_method=greedy_selection, save_fig=True, random_optimalism=True, specific_vid_name="inference_seq", with_dict=True)
			else:
				sequence = agent.generate_observation(action_selection_method=greedy_selection, save_fig=True, random_optimalism=True, specific_vid_name="inference_seq")
				recognizer.dump_plans(true_sequence=sequence, true_goal=goal, percentage=percentage)
			partial_sequence = random_subset_with_order(sequence, (int)(percentage * len(sequence)), is_fragmented=args.partial_obs_type=="fragmented")
			# add evaluation_function to kwargs if this is graql. move everything to kwargs...
			closest_goal = recognizer.inference_phase(partial_sequence, goal, percentage)
			# print(f'real goal {goal}, closest goal is: {closest_goal}')
			if all(a == b for a, b in zip(goal, closest_goal)):
				correct += 1
			task_num += 1

	print(f'correct: {correct}\n total tasks: {task_num}\n accuracy: {correct/task_num}')

def parse_args():
	parser = argparse.ArgumentParser(
		description="Parse command-line arguments for the RL experiment.",
		formatter_class=argparse.RawTextHelpFormatter
	)

	# Required arguments
	required_group = parser.add_argument_group("Required arguments")
	required_group.add_argument("--domain", choices=["point_maze", "minigrid", "parking", "franka_kitchen"], required=True, help="Domain type (point_maze, minigrid, parking, or franka_kitchen)")
	required_group.add_argument("--recognizer", choices=["graml", "graql", "draco"], required=True, help="Recognizer type (graml, graql, draco). graql only for discrete domains.")
	required_group.add_argument("--task", choices=["L1", "L2", "L3"], required=True, help="Task identifier (e.g., L1, L2,...,L5)")
	required_group.add_argument("--partial_obs_type", required=True, choices=["fragmented", "continuing"], help="Give fragmented or continuing partial observations for inference phase inputs.")

	# Optional arguments
	optional_group = parser.add_argument_group("Optional arguments")
	optional_group.add_argument("--collect_stats", action="store_true", help="Whether to collect statistics")
	optional_group.add_argument("--minigrid_env", choices=["four_rooms", "obstacles"], help="Minigrid environment (four_rooms or obstacles)")
	optional_group.add_argument("--parking_env", choices=["agent", "gc_agent"], help="Parking environment (agent or gc_agent)")
	optional_group.add_argument("--point_maze_env", choices=["obstacles", "four_rooms"], help="Parking environment (agent or gc_agent)")
	optional_group.add_argument("--franka_env", choices=["comb1", "comb2"], help="Franka Kitchen environment (comb1 or comb2)")
	optional_group.add_argument("--learn_same_seq_len", help="Learn with the same sequence length")
	optional_group.add_argument("--inference_same_seq_len", help="Infer with the same sequence length")

	args = parser.parse_args()
 
	### VALIDATE INPUTS ###
	# Assert that all required arguments are provided
	assert args.domain is not None and args.recognizer is not None and args.task is not None, "Missing required arguments: domain, recognizer, or task"

	 # Validate the combination of domain and environment
	if args.domain == "minigrid" and args.minigrid_env is None:
		parser.error("Missing required argument: --minigrid_env must be provided when --domain is minigrid")
	elif args.domain == "parking" and args.parking_env is None:
		parser.error("Missing required argument: --parking_env must be provided when --domain is parking")
	elif args.domain == "point_maze" and args.point_maze_env is None:
		parser.error("Missing required argument: --point_maze_env must be provided when --domain is point_maze")
	elif args.domain == "franka_kitchen" and args.franka_env is None:
		parser.error("Missing required argument: --franka_env must be provided when --domain is franka_kitchen")

	# Set default values for optional arguments if not provided and assert they're only given for graml
	if args.recognizer != "graml":
		if args.learn_same_seq_len != None: parser.error("learn_same_seq_len is only relevant for graml.")
		if args.inference_same_seq_len != None: parser.error("inference_same_seq_len is only relevant for graml.")
	else:
		if args.learn_same_seq_len == None: args.learn_same_seq_len = False
		if args.inference_same_seq_len == None: args.inference_same_seq_len = False

	return args

if __name__ == "__main__":
	# assert (len(sys.argv) == 7 and sys.argv[1] in ["graml"] and sys.argv[2] in ["continuing_partial_obs", "fragmented_partial_obs"] and sys.argv[3] in ["inference_same_length", "inference_diff_length"] and sys.argv[4] in ["learn_same_length", "learn_diff_length"] and sys.argv[5] in ['no_collect_statistics', 'collect_statistics'] and sys.argv[6] in ["MAZE:FOUR_ROOMS", "MAZE:OBSTACLES", "MINIGRID", "PARKING:GC_AGENT", "PARKING:AGENT", "KITCHEN:COMB1"]) \
	# 		or (len(sys.argv) == 5 and sys.argv[1] in ["graql"] and sys.argv[2] in ["continuing_partial_obs", "fragmented_partial_obs"] and sys.argv[3] in ['no_collect_statistics', 'collect_statistics'] and sys.argv[4] in ["MAZE:FOUR_ROOMS", "MAZE:OBSTACLES", "MINIGRID", "PARKING:GC_AGENT", "PARKING:AGENT", "KITCHEN:COMB1"]) \
	#    ,f"Assertion failed: incorrect arguments.\nExample 1: \n\t python graml_main.py graml [continuing_partial_obs/fragmented_partial_obs] [inference_same_length/inference_diff_length] [learn_same_length/learn_diff_length] [collect_statistics/no_collect_statistics]\nExample 2: \n\t python graml_main.py graql [continuing_partial_obs/fragmented_partial_obs] [collect_statistics/no_collect_statistics]"
	# if sys.argv[1] == "graml":
	# 	set_global_storage_configs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	# 	init(recognizer_str=sys.argv[1], is_fragmented=sys.argv[2] == "fragmented_partial_obs", is_inference_same_length_sequences=sys.argv[3] == "inference_same_length", is_learn_same_length_sequences=sys.argv[4] == "learn_same_length", collect_statistics=sys.argv[5] == "collect_statistics", task=sys.argv[6])
	# else: # graql
	# 	set_global_storage_configs(sys.argv[1], sys.argv[2])
	# 	init(recognizer_str=sys.argv[1], is_fragmented=sys.argv[2] == "fragmented_partial_obs", collect_statistics=sys.argv[3] == "collect_statistics", task=sys.argv[4])
	args = parse_args()
	set_global_storage_configs(recognizer_str=args.recognizer, is_fragmented=args.partial_obs_type, is_inference_same_length_sequences=args.inference_same_seq_len, is_learn_same_length_sequences=args.learn_same_seq_len)
	run_experiments(args)

	
# python experiments.py --recognizer graml --domain point_maze --task L1 --partial_obs_type fragmented --point_maze_env obstacles --inference_same_seq_len --collect_stats
# python experiments.py --recognizer graql --domain point_maze --task L1 --partial_obs_type fragmented --point_maze_env obstacles --collect_stats