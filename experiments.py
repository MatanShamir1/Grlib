import argparse
import time
import dill

from stable_baselines3 import PPO

from grlib.metrics import metrics
from grlib.ml.neural.SB3model import NeuralAgent
from grlib.recognizer.graml.graml_recognizer import MCTS_BASED, AGENT_BASED, GC_AGENT_BASED
from grlib.recognizer.graql.graql_recognizer import GraqlRecognizer
from grlib.recognizer import GramlRecognizer
from grlib.metrics.metrics import stochastic_amplified_selection
from grlib.ml.utils.format import random_subset_with_order
from grlib.recognizer.utils import recognizer_str_to_obj
from grlib.ml.utils.storage import create_folders_if_necessary, get_experiment_results_path, set_global_storage_configs

# keep this import last, dependent on lots of initialized modules.
from consts import PROBLEMS

def run_experiments(args):

	# kwargs for every recognizer:
	recognizer_type = recognizer_str_to_obj(args.recognizer)
	recognizer_kwargs = {}

	# relevant also for inference phase experiments
	domain_inputs = PROBLEMS[args.domain]
	env_name, = [x for x in [args.minigrid_env, args.parking_env, args.point_maze_env, args.franka_env, args.panda_env] if isinstance(x, str)]
	task_inputs = domain_inputs["tasks"][env_name]
	task_str_to_goal = task_inputs[args.task]["task_str_to_goal"]
	learner_type = domain_inputs["additional_combined_recognizer_kwargs"]["learner_type"]
	dynamic_goals_problems, dynamic_train_configs = zip(*task_inputs[args.task]["dynamic_goals_train_configs"])
	if "use_goal_directed_problem" in domain_inputs["additional_combined_recognizer_kwargs"].keys() and env_name == 'gc_agent':
		recognizer_kwargs["use_goal_directed_problem"] = domain_inputs["additional_combined_recognizer_kwargs"]["use_goal_directed_problem"]

	# relevant for all recognizers' inputs
	recognizer_kwargs["env_name"] = args.domain
	base_problems, base_train_configs = zip(*task_inputs[args.task]["base_goals_problems_train_configs"])
	recognizer_kwargs["problems"] = base_problems
	recognizer_kwargs["task_str_to_goal"] = task_str_to_goal
	recognizer_kwargs["method"] = learner_type
	#then make sure parking-v0 PPO version of ben is correctly loaded.
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
	start_dlp_time = time.time()
	recognizer.domain_learning_phase()
	dlp_time = time.time() - start_dlp_time
	start_ga_time = time.time()
	recognizer.goals_adaptation_phase(dynamic_goals_problems=dynamic_goals_problems, dynamic_train_configs=tuple((config[0], config[2]) for config in dynamic_train_configs))
	ga_time = time.time() - start_ga_time
	# experiments
	total_tasks, total_correct, results = 0, 0, {}
	for percentage in [0.3, 0.5, 0.7, 0.9, 1]:
		curr_num_tasks, curr_correct, curr_sum_inf_time = 0, 0, 0
		for problem, dynamic_train_config in zip(dynamic_goals_problems, dynamic_train_configs):
			goal = str(task_str_to_goal(problem))
			start_inf_time = time.time()
			kwargs = {"env_name":args.domain, "problem_name":problem}
			if dynamic_train_config[1]: kwargs["algorithm"] = dynamic_train_config[1]
			if dynamic_train_config[2]: kwargs["num_timesteps"] = dynamic_train_config[2]
			agent = learner_type(**kwargs)
			agent.learn()
			generate_obs_kwargs = {"action_selection_method": stochastic_amplified_selection,
                          			"save_fig": True,
                             		"random_optimalism": True,
                               		"specific_vid_name": "inference_seq"}

			# need to dump the whole plan for draco because it needs it for inference phase for checking likelihood.
			if recognizer_type == GraqlRecognizer and learner_type == NeuralAgent:
				generate_obs_kwargs["with_dict"] = True

			sequence = agent.generate_observation(**generate_obs_kwargs)
			if recognizer_type == GramlRecognizer: # need to dump the plans to compute offline plan similarity only in graml's case for evaluation.
				recognizer.dump_plans(true_sequence=sequence, true_goal=goal, percentage=percentage)
			partial_sequence = random_subset_with_order(sequence, (int)(percentage * len(sequence)), is_fragmented=args.partial_obs_type=="fragmented")
			# add evaluation_function to kwargs if this is graql. move everything to kwargs...
			closest_goal = recognizer.inference_phase(partial_sequence, goal, percentage)
			# print(f'real goal {goal}, closest goal is: {closest_goal}')
			if all(a == b for a, b in zip(goal, closest_goal)):
				curr_correct += 1
			curr_num_tasks += 1
			curr_sum_inf_time += (time.time() - start_inf_time)
		total_tasks += curr_num_tasks
		total_correct += curr_correct
		results[str(percentage)] = {'correct': curr_correct, 'num_of_tasks': curr_num_tasks, 'accuracy': curr_correct/curr_num_tasks, 'average_inference_time': curr_sum_inf_time/curr_num_tasks}

	total_average_inference_time = sum([result['average_inference_time'] for result in results.values()]) / len(results)
	results['total'] = {'total_correct': total_correct, 'total_tasks': total_tasks, "total_accuracy": total_correct/total_tasks, 'total_average_inference_time': total_average_inference_time, 'goals_adaptation_time': ga_time, 'domain_learning_time': dlp_time}
	print(str(results))
	res_file_path = get_experiment_results_path(args.domain, env_name, args.task)
	create_folders_if_necessary(res_file_path)
	print(f"generating results into {res_file_path}")
	with open(f'{res_file_path}.pkl', 'wb') as results_file:
		dill.dump(results, results_file)
	with open(f'{res_file_path}.txt', 'w') as results_file:
		results_file.write(str(results))


def parse_args():
	parser = argparse.ArgumentParser(
		description="Parse command-line arguments for the RL experiment.",
		formatter_class=argparse.RawTextHelpFormatter
	)

	# Required arguments
	required_group = parser.add_argument_group("Required arguments")
	required_group.add_argument("--domain", choices=["point_maze", "minigrid", "parking", "franka_kitchen", "panda"], required=True, help="Domain type (point_maze, minigrid, parking, or franka_kitchen)")
	required_group.add_argument("--recognizer", choices=["graml", "graql", "draco"], required=True, help="Recognizer type (graml, graql, draco). graql only for discrete domains.")
	required_group.add_argument("--task", choices=["L1", "L2", "L3", "L4", "L5", "L11", "L22", "L33", "L44", "L55", "L111", "L222", "L333", "L444", "L555"], required=True, help="Task identifier (e.g., L1, L2,...,L5)")
	required_group.add_argument("--partial_obs_type", required=True, choices=["fragmented", "continuing"], help="Give fragmented or continuing partial observations for inference phase inputs.")

	# Optional arguments
	optional_group = parser.add_argument_group("Optional arguments")
	optional_group.add_argument("--collect_stats", action="store_true", help="Whether to collect statistics")
	optional_group.add_argument("--minigrid_env", choices=["lava_crossing", "obstacles"], help="Minigrid environment (four_rooms or obstacles)")
	optional_group.add_argument("--parking_env", choices=["gd_agent", "gc_agent"], help="Parking environment (agent or gc_agent)")
	optional_group.add_argument("--point_maze_env", choices=["obstacles", "four_rooms"], help="Parking environment (agent or gc_agent)")
	optional_group.add_argument("--franka_env", choices=["comb1", "comb2"], help="Franka Kitchen environment (comb1 or comb2)")
	optional_group.add_argument("--panda_env", choices=["gc_agent", "gd_agent"], help="Panda Robotics environment (gc_agent or gd_agent)")
	optional_group.add_argument("--learn_same_seq_len", action="store_true", help="Learn with the same sequence length")
	optional_group.add_argument("--inference_same_seq_len", action="store_true", help="Infer with the same sequence length")

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
	elif args.domain == "panda" and args.panda_env is None:
		parser.error("Missing required argument: --panda_env must be provided when --domain is panda")

	# Set default values for optional arguments if not provided and assert they're only given for graml
	if args.recognizer != "graml":
		if args.learn_same_seq_len == True: parser.error("learn_same_seq_len is only relevant for graml.")
		if args.inference_same_seq_len == True: parser.error("inference_same_seq_len is only relevant for graml.")

	return args

if __name__ == "__main__":
	args = parse_args()
	set_global_storage_configs(recognizer_str=args.recognizer, is_fragmented=args.partial_obs_type, is_inference_same_length_sequences=args.inference_same_seq_len, is_learn_same_length_sequences=args.learn_same_seq_len)
	run_experiments(args)
