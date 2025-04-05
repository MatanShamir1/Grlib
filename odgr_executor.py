import argparse
import os
import time
import dill

from gr_libs.environment.utils.utils import domain_to_env_property
from gr_libs.ml.neural.deep_rl_learner import DeepRLAgent
from gr_libs.recognizer.gr_as_rl.gr_as_rl_recognizer import Draco, GCDraco, Graql
from gr_libs.recognizer import Graml
from gr_libs.metrics.metrics import stochastic_amplified_selection
from gr_libs.ml.utils.format import random_subset_with_order
from gr_libs.recognizer.recognizer import GaAgentTrainerRecognizer, LearningRecognizer
from gr_libs.recognizer.utils import recognizer_str_to_obj
from gr_libs.ml.utils.storage import create_folders_if_necessary, get_and_create, get_experiment_results_path, get_policy_sequences_result_path

from gr_libs.problems.consts import PROBLEMS

def validate(args, recognizer_type, task_inputs):
	if "base" in task_inputs.keys():
		#assert issubclass(recognizer_type, LearningRecognizer), f"base is in the task_inputs for the recognizer {args.recognizer}, which doesn't have a domain learning phase (is not a learning recognizer)."
		assert list(task_inputs.keys())[0] == "base", "In case of LearningRecognizer, base should be the first element in the task_inputs dict in consts.py"
		assert "base" not in list(task_inputs.keys())[1:], "In case of LearningRecognizer, base should be only in the first element in the task_inputs dict in consts.py"
	#else:
		#assert not issubclass(recognizer_type, LearningRecognizer), f"base is not in the task_inputs for the recognizer {args.recognizer}, which has a domain learning phase (is a learning recognizer). Remove it from the task_inputs dict in consts.py."

def run_odgr_problem(args):
	recognizer_type = recognizer_str_to_obj(args.recognizer)
	env_inputs = PROBLEMS[args.domain]
	assert args.env_name in env_inputs.keys(), f"env_name {args.env_name} is not in the list of available environments for the domain {args.domain}. Add it to PROBLEMS dict in consts.py"
	task_inputs = env_inputs[args.env_name][args.task]
	recognizer = recognizer_type(domain_name=args.domain, env_name=args.env_name, collect_statistics=args.collect_stats)
	validate(args, recognizer_type, task_inputs)
	ga_times, results = [], {}
	for key, value in task_inputs.items():
		if key == "base":
			dlp_time = 0
			if issubclass(recognizer_type, LearningRecognizer):
				start_dlp_time = time.time()
				recognizer.domain_learning_phase(base_goals=value["goals"], train_configs=value["train_configs"])
				dlp_time = time.time() - start_dlp_time
		elif key.startswith("G_"):
			start_ga_time = time.time()
			kwargs = {"dynamic_goals": value["goals"]}
			if issubclass(recognizer_type, GaAgentTrainerRecognizer): kwargs["dynamic_train_configs"] = value["train_configs"]
			recognizer.goals_adaptation_phase(**kwargs)
			ga_times.append(time.time() - start_ga_time)
		elif key.startswith("I_"):
			goal, train_config, consecutive, consecutive_str, percentage = value["goal"], value["train_config"], value["consecutive"], "consecutive" if value["consecutive"] == True else "non_consecutive", value["percentage"]
			results.setdefault(str(percentage), {})
			results[str(percentage)].setdefault(consecutive_str, {'correct': 0, 'num_of_tasks': 0, 'accuracy': 0, 'average_inference_time': 0})
			property_type = domain_to_env_property(args.domain)
			env_property = property_type(args.env_name)
			problem_name = env_property.goal_to_problem_str(goal)
			rl_agent_type = recognizer.rl_agent_type
			agent = rl_agent_type(domain_name=args.domain, problem_name=problem_name, algorithm=train_config[0], num_timesteps=train_config[1])
			agent.learn()
			fig_path = get_and_create(f"{os.path.abspath(os.path.join(get_policy_sequences_result_path(domain_name=args.domain, env_name=args.env_name, recognizer=args.recognizer), problem_name))}_inference_seq")
			generate_obs_kwargs = {"action_selection_method": stochastic_amplified_selection,
									"save_fig": True,
									"random_optimalism": True,
									"fig_path": fig_path,
									"env_prop": env_property}

			# need to dump the whole plan for draco because it needs it for inference phase for checking likelihood.
			if (recognizer_type == Draco or recognizer_type == GCDraco) and issubclass(rl_agent_type, DeepRLAgent): # TODO remove this condition, remove the assumption.
				generate_obs_kwargs["with_dict"] = True
			sequence = agent.generate_observation(**generate_obs_kwargs)
			if issubclass(recognizer_type, Graml): # need to dump the plans to compute offline plan similarity only in graml's case for evaluation.
				recognizer.dump_plans(true_sequence=sequence, true_goal=goal, percentage=percentage)
			partial_sequence = random_subset_with_order(sequence, (int)(percentage * len(sequence)), is_consecutive=consecutive)
			# add evaluation_function to kwargs if this is graql. move everything to kwargs...
			start_inf_time = time.time()
			closest_goal = recognizer.inference_phase(partial_sequence, goal, percentage)
			results[str(percentage)][consecutive_str]['average_inference_time'] += time.time() - start_inf_time
			# print(f'real goal {goal}, closest goal is: {closest_goal}')
			if all(a == b for a, b in zip(str(goal), closest_goal)):
				results[str(percentage)][consecutive_str]['correct'] += 1
			results[str(percentage)][consecutive_str]['num_of_tasks'] += 1

	for percentage in results.keys():
		for consecutive_str in results[str(percentage)].keys():
			results[str(percentage)][consecutive_str]['average_inference_time'] /= len(results[str(percentage)][consecutive_str])
			results[str(percentage)][consecutive_str]['accuracy'] = results[str(percentage)][consecutive_str]['correct'] / results[str(percentage)][consecutive_str]['num_of_tasks']
	
	# aggregate
	total_correct = sum([result['correct'] for cons_result in results.values() for result in cons_result.values()])
	total_tasks	= sum([result['num_of_tasks'] for cons_result in results.values() for result in cons_result.values()])
	total_average_inference_time = sum([result['average_inference_time'] for cons_result in results.values() for result in cons_result.values()]) / total_tasks

	results['total'] = {'total_correct': total_correct, 'total_tasks': total_tasks, "total_accuracy": total_correct/total_tasks, 'total_average_inference_time': total_average_inference_time, 'goals_adaptation_time': sum(ga_times)/len(ga_times), 'domain_learning_time': dlp_time}
	print(str(results))
	res_file_path = get_and_create(get_experiment_results_path(domain=args.domain, env_name=args.env_name, task=args.task, recognizer=args.recognizer))
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
	required_group.add_argument("--domain", choices=["point_maze", "minigrid", "parking", "panda"], required=True, help="Domain name (point_maze, minigrid, parking, or panda)")
	required_group.add_argument("--env_name", required=True, help="Env name (point_maze, minigrid, parking, or panda). For example, Parking-S-14-PC--v0")
	required_group.add_argument("--recognizer", choices=["MCTSBasedGraml", "ExpertBasedGraml", "GCGraml", "Graql", "Draco", "GCDraco"], required=True, help="Recognizer type. Follow readme.md and recognizer folder for more information and rules.")
	required_group.add_argument("--task", choices=["L1", "L2", "L3", "L4", "L5", "L11", "L22", "L33", "L44", "L55", "L111", "L222", "L333", "L444", "L555"], required=True, help="Task identifier (e.g., L1, L2,...,L5)")

	# Optional arguments
	optional_group = parser.add_argument_group("Optional arguments")
	optional_group.add_argument("--collect_stats", action="store_true", help="Whether to collect statistics")
	args = parser.parse_args()
 
	### VALIDATE INPUTS ###
	# Assert that all required arguments are provided
	assert args.domain is not None and args.recognizer is not None and args.task is not None, "Missing required arguments: domain, recognizer, or task"
	return args

if __name__ == "__main__":
	args = parse_args()
	run_odgr_problem(args)
