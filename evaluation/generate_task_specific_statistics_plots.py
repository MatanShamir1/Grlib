import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import inspect
import torch
import dill

from gr_libs.ml.utils import get_embeddings_result_path
from gr_libs.ml.utils.storage import get_experiment_results_path, set_global_storage_configs, get_graql_experiment_confidence_path
from gr_libs.metrics.metrics import measure_average_sequence_distance

def get_tasks_embeddings_dir_path(env_name):
	return os.path.join("../gr_libs", get_embeddings_result_path(env_name))

def get_figures_dir_path(domain_name, env_name):
	return os.path.join("../gr_libs", "figures", domain_name, env_name)

def similarities_vector_to_std_deviation_units_vector(ref_dict: dict, relative_to_largest):
	"""
	Calculate the number of standard deviation units every other element is 
	from the largest/smallest element in the vector.
	
	Parameters:
	- vector: list or numpy array of numbers.
	- relative_to_largest: boolean, if True, measure in relation to the largest element,
						   if False, measure in relation to the smallest element.
	
	Returns:
	- List of number of standard deviation units for each element in the vector.
	"""
	vector = np.array(list(ref_dict.values()))
	mean = np.mean(vector) # for the future maybe another method for measurement
	std_dev = np.std(vector)
	
	# Determine the reference element (largest or smallest)
	if relative_to_largest:
		reference_value = np.max(vector)
	else:
		reference_value = np.min(vector)
	for goal, value in ref_dict.items():
		ref_dict[goal] = abs(value - reference_value) / std_dev
	return ref_dict

def analyze_and_produce_plots(recognizer_type: str, domain_name: str, env_name: str, fragmented_status: str, inf_same_length_status: str, learn_same_length_status: str):
	if recognizer_type == "graml":
		assert os.path.exists(get_embeddings_result_path(domain_name)), "Embeddings weren't made for this environment, run graml_main.py with this environment first."
		tasks_embedding_dicts = {}
		tasks_plans_dict = {}
		goals_similarity_dict = {}
		plans_similarity_dict = {}

		embeddings_dir_path = get_tasks_embeddings_dir_path(domain_name)
		for embeddings_file_name in [filename for filename in os.listdir(embeddings_dir_path) if 'embeddings' in filename]:
			with open(os.path.join(embeddings_dir_path, embeddings_file_name), 'rb') as emb_file:
				splitted_name = embeddings_file_name.split('_')
				goal, percentage = splitted_name[0], splitted_name[1]
				with open(os.path.join(embeddings_dir_path, f'{goal}_{percentage}_plans_dict.pkl'), 'rb') as plan_file:
					tasks_plans_dict[f"{goal}_{percentage}"] = dill.load(plan_file)
				tasks_embedding_dicts[f"{goal}_{percentage}"] = dill.load(emb_file)
	
		for goal_percentage, embedding_dict in tasks_embedding_dicts.items():
			goal, percentage = goal_percentage.split('_')
			similarities = {dynamic_goal: [] for dynamic_goal in embedding_dict.keys() if 'true' not in dynamic_goal}
			real_goal_embedding = embedding_dict[f"{goal}_true"]
			for dynamic_goal, goal_embedding in embedding_dict.items():
				if 'true' in dynamic_goal: continue
				curr_similarity = torch.exp(-torch.sum(torch.abs(goal_embedding-real_goal_embedding)))
				similarities[dynamic_goal] = curr_similarity.item()
			if goal not in goals_similarity_dict.keys(): goals_similarity_dict[goal] = {}
			goals_similarity_dict[goal][percentage] = similarities_vector_to_std_deviation_units_vector(ref_dict=similarities, relative_to_largest=True)
   
		for goal_percentage, plans_dict in tasks_plans_dict.items():
			goal, percentage = goal_percentage.split('_')
			real_plan = plans_dict[f"{goal}_true"]
			sequence_similarities = {d_goal:measure_average_sequence_distance(real_plan, plan) for d_goal,plan in plans_dict.items() if 'true' not in d_goal} # aps = agent plan sequence?
			if goal not in plans_similarity_dict.keys(): plans_similarity_dict[goal] = {}
			plans_similarity_dict[goal][percentage] = similarities_vector_to_std_deviation_units_vector(ref_dict=sequence_similarities, relative_to_largest=False)

		goals = list(goals_similarity_dict.keys())
		percentages = sorted(set(percentage for similarities in goals_similarity_dict.values() for percentage in similarities.keys()))
		num_percentages = len(percentages)
		fig_string = f"{recognizer_type}_{domain_name}_{env_name}_{fragmented_status}_{inf_same_length_status}_{learn_same_length_status}"

	else: # algorithm = "graql"
		assert os.path.exists(get_graql_experiment_confidence_path(domain_name)), "Embeddings weren't made for this environment, run graml_main.py with this environment first."
		tasks_scores_dict = {}
		goals_similarity_dict = {}
		experiments_dir_path = get_graql_experiment_confidence_path(domain_name)
		for experiments_file_name in os.listdir(experiments_dir_path):
			with open(os.path.join(experiments_dir_path, experiments_file_name), 'rb') as exp_file:
				splitted_name = experiments_file_name.split('_')
				goal, percentage = splitted_name[1], splitted_name[2]
				tasks_scores_dict[f"{goal}_{percentage}"] = dill.load(exp_file)

		for goal_percentage, scores_list in tasks_scores_dict.items():
			goal, percentage = goal_percentage.split('_')
			similarities = {dynamic_goal: score for (dynamic_goal, score) in scores_list}
			if goal not in goals_similarity_dict.keys(): goals_similarity_dict[goal] = {}
			goals_similarity_dict[goal][percentage] = similarities_vector_to_std_deviation_units_vector(ref_dict=similarities, relative_to_largest=False)
	
		goals = list(goals_similarity_dict.keys())
		percentages = sorted(set(percentage for similarities in goals_similarity_dict.values() for percentage in similarities.keys()))
		num_percentages = len(percentages)
		fig_string = f"{recognizer_type}_{domain_name}_{env_name}_{fragmented_status}"

	# -------------------- Start of Confusion Matrix Code --------------------
	# Initialize matrices of size len(goals) x len(goals)
	confusion_matrix_goals, confusion_matrix_plans = np.zeros((len(goals), len(goals))), np.zeros((len(goals), len(goals)))

	# if domain_name == 'point_maze' and args.task == 'L555':
	# 	if env_name == 'obstacles':
	# 		goals = ['(4, 7)', '(3, 6)', '(5, 5)', '(8, 8)', '(6, 3)', '(7, 4)']
	# 	else: # if env_name is 'four_rooms'
	# 		goals = ['(2, 8)', '(3, 7)', '(3, 4)', '(4, 4)', '(4, 3)', '(7, 3)', '(8, 2)']

	# Populate confusion matrix with similarity values for goals
	for i, true_goal in enumerate(goals):
		for j, dynamic_goal in enumerate(goals):
			percentage = percentages[-3]
			confusion_matrix_goals[i, j] = goals_similarity_dict[true_goal][percentage].get(dynamic_goal, 0)

	if plans_similarity_dict:
		# Populate confusion matrix with similarity values for plans
		for i, true_goal in enumerate(goals):
			for j, dynamic_goal in enumerate(goals):
				percentage = percentages[-1]
				confusion_matrix_plans[i, j] = plans_similarity_dict[true_goal][percentage].get(dynamic_goal, 0)

	# Create the figure and subplots for the unified display
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

	# Plot for goal similarities
	im1 = ax1.imshow(confusion_matrix_goals, cmap='Blues', interpolation='nearest')
	cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
	cbar1.set_label('St. dev from most probable goal', fontsize=18)
	ax1.set_title('Embeddings', fontsize=22, pad=20)
	ax1.set_xticks(np.arange(len(goals)))
	ax1.set_xticklabels(goals, rotation=45, ha="right", fontsize=16)
	ax1.set_yticks(np.arange(len(goals)))
	ax1.set_yticklabels(goals, fontsize=16)  # y-tick labels for ax1

	# Plot for plan similarities
	im2 = ax2.imshow(confusion_matrix_plans, cmap='Greens', interpolation='nearest')
	cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
	cbar2.set_label('Distance between plans', fontsize=18)
	ax2.set_title('Sequences', fontsize=22, pad=20)
	ax2.set_xticks(np.arange(len(goals)))
	ax2.set_xticklabels(goals, rotation=45, ha="right", fontsize=16)
	ax2.set_yticks(np.arange(len(goals)))  # y-ticks for ax2 explicitly
	ax2.set_yticklabels(goals, fontsize=16)  # y-tick labels for ax2

	# Adjust the figure layout to reduce overlap
	plt.subplots_adjust(left=0.15, right=0.9, bottom=0.25, top=0.85, wspace=0.1)

	# Unified axis labels, placed closer to the left
	fig.text(0.57, 0.07, 'Goals Adaptation Phase', ha='center', fontsize=22)
	fig.text(0.12, 0.5, 'Inference Phase', va='center', rotation='vertical', fontsize=22)

	# Save the combined plot
	fig_dir = get_figures_dir_path(domain_name=domain_name, env_name=env_name)
	if not os.path.exists(fig_dir):
		os.makedirs(fig_dir)
	confusion_matrix_combined_path = os.path.join(fig_dir, f"{fig_string}_combined_conf_mat.png")
	plt.savefig(confusion_matrix_combined_path, dpi=300)
	print(f"Combined confusion matrix figure saved at: {confusion_matrix_combined_path}")

	# -------------------- End of Confusion Matrix Code --------------------
	fig, axes = plt.subplots(nrows=num_percentages, ncols=1, figsize=(10, 6 * num_percentages))

	if num_percentages == 1:
		axes = [axes]

	for i, percentage in enumerate(percentages):
		correct_tasks, tasks_num = 0, 0
		ax = axes[i]
		dynamic_goals = list(next(iter(goals_similarity_dict.values()))[percentage].keys())
		num_goals = len(goals)
		num_dynamic_goals = len(dynamic_goals)
		bar_width = 0.8 / num_dynamic_goals
		bar_positions = np.arange(num_goals)

		if recognizer_type == "graml":
			for j, dynamic_goal in enumerate(dynamic_goals):
				goal_similarities = [goals_similarity_dict[goal][percentage][dynamic_goal] + 0.04 for goal in goals]
				plan_similarities = [plans_similarity_dict[goal][percentage][dynamic_goal] + 0.04 for goal in goals]
				ax.bar(bar_positions + j * bar_width, goal_similarities, bar_width/2, label=f"embedding of {dynamic_goal}")
				ax.bar(bar_positions + j * bar_width + bar_width/2, plan_similarities, bar_width/2, label=f"plan to {dynamic_goal}")
		else:
				for j, dynamic_goal in enumerate(dynamic_goals):
					goal_similarities = [goals_similarity_dict[goal][percentage][dynamic_goal] + 0.04 for goal in goals]
					ax.bar(bar_positions + j * bar_width, goal_similarities, bar_width, label=f"policy to {dynamic_goal}")

		x_labels = []
		for true_goal in goals:
			guessed_goal = min(goals_similarity_dict[true_goal][percentage], key=goals_similarity_dict[true_goal][percentage].get)
			tasks_num += 1
			if true_goal == guessed_goal: correct_tasks += 1
			second_lowest_value = sorted(goals_similarity_dict[true_goal][percentage].values())[1]
			confidence_level = abs(goals_similarity_dict[true_goal][percentage][guessed_goal] - second_lowest_value)
			label = f"True: {true_goal}\nGuessed: {guessed_goal}\nConfidence: {confidence_level:.2f}"
			x_labels.append(label)

		ax.set_ylabel('Distance (units in st. deviations)', fontsize=10)
		ax.set_title(f'Confidence level for {domain_name}, {env_name}, {fragmented_status}. Accuracy: {correct_tasks / tasks_num}', fontsize=12)
		ax.set_xticks(bar_positions + bar_width * (num_dynamic_goals - 1) / 2)
		ax.set_xticklabels(x_labels, fontsize=8)
		ax.legend()

	fig_path = os.path.join(fig_dir, f"{fig_string}_stats.png")
	fig.savefig(fig_path)
	print(f"general figure saved at: {fig_path}")

 
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
	optional_group.add_argument("--minigrid_env", choices=["four_rooms", "obstacles"], help="Minigrid environment (four_rooms or obstacles)")
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

	if args.recognizer != "graml":
		if args.learn_same_seq_len == True: parser.error("learn_same_seq_len is only relevant for graml.")
		if args.inference_same_seq_len == True: parser.error("inference_same_seq_len is only relevant for graml.")

	return args

if __name__ == "__main__":
	args = parse_args()
	set_global_storage_configs(recognizer_str=args.recognizer, is_fragmented=args.partial_obs_type,
							is_inference_same_length_sequences=args.inference_same_seq_len, is_learn_same_length_sequences=args.learn_same_seq_len)
	env_name, = [x for x in [args.minigrid_env, args.parking_env, args.point_maze_env, args.franka_env] if isinstance(x, str)]
	if args.inference_same_seq_len: inference_same_seq_len = "inference_same_seq_len"
	else: inference_same_seq_len = "inference_diff_seq_len"
	if args.learn_same_seq_len: learn_same_seq_len = "learn_same_seq_len"
	else: learn_same_seq_len = "learn_diff_seq_len"
	analyze_and_produce_plots(args.recognizer, domain_name=args.domain, env_name=env_name, fragmented_status=args.partial_obs_type,
						   inf_same_length_status=inference_same_seq_len, learn_same_length_status=learn_same_seq_len)