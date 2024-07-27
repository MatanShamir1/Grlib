import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import inspect
import torch
import dill

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
GRAML_itself = os.path.dirname(currentdir)
GRAML_includer = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, GRAML_includer)
sys.path.insert(0, GRAML_itself)

from GRAML.ml.utils import get_embeddings_result_path
from GRAML.ml.utils.storage import set_global_storage_configs, get_graql_experiment_confidence_path

def get_tasks_embeddings_dir_path(env_name):
	return GRAML_itself + '/' + get_embeddings_result_path(env_name)

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

def analyze_and_produce_plots(algorithm: str, confs: list[str]):
	if algorithm == "graml":
		env_name, fragmented_status, inf_same_length_status, learn_same_length_status = confs[0], confs[1], confs[2], confs[3]
		set_global_storage_configs(algorithm, fragmented_status, inf_same_length_status, learn_same_length_status)
		assert os.path.exists(get_embeddings_result_path(confs[0])), "Embeddings weren't made for this environment, run graml_main.py with this environment first."
		tasks_embedding_dicts = {}
		goals_similarity_dict = {}

		embeddings_dir_path = get_tasks_embeddings_dir_path(env_name)
		for embeddings_file_name in os.listdir(embeddings_dir_path):
			with open(os.path.join(embeddings_dir_path, embeddings_file_name), 'rb') as emb_file:
				splitted_name = embeddings_file_name.split('_')
				goal, percentage = splitted_name[0], splitted_name[1]
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
   
		goals = list(goals_similarity_dict.keys())
		percentages = sorted(set(percentage for similarities in goals_similarity_dict.values() for percentage in similarities.keys()))
		num_percentages = len(percentages)
		fig_string = f"{algorithm}_{env_name}_{fragmented_status}_{inf_same_length_status}_{learn_same_length_status}"

	else: # algorithm = "graql"
		env_name, fragmented_status = confs[0], confs[1]
		set_global_storage_configs(algorithm, fragmented_status)
		assert os.path.exists(get_graql_experiment_confidence_path(env_name)), "Embeddings weren't made for this environment, run graml_main.py with this environment first."
		tasks_scores_dict = {}
		goals_similarity_dict = {}
		experiments_dir_path = get_graql_experiment_confidence_path(env_name)
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
		fig_string = f"{algorithm}_{env_name}_{fragmented_status}"
	
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

		for j, dynamic_goal in enumerate(dynamic_goals):
			similarities = [goals_similarity_dict[goal][percentage][dynamic_goal] for goal in goals]
			ax.bar(bar_positions + j * bar_width, similarities, bar_width, label=f"{dynamic_goal}")

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
		ax.set_title(f'Confidence level for {env_name}, {fragmented_status}. Accuracy: {correct_tasks / tasks_num}', fontsize=12)
		ax.set_xticks(bar_positions + bar_width * (num_dynamic_goals - 1) / 2)
		ax.set_xticklabels(x_labels, fontsize=8)
		ax.legend()

	# Save the figure
	fig.savefig(f"{fig_string}_goal_similarities_combined.png")
	print(f"figure saved at: {os.path.curdir}/{fig_string}_goal_similarities_combined.png")

	# Show plot
	plt.tight_layout()
	plt.show()
	
if __name__ == "__main__":
	# checks:
	assert len(sys.argv) == 3 and sys.argv[1] in ["graql", "graml"], f"Assertion failed: len(sys.argv) is {len(sys.argv)} while it needs to be 3.\n Example 1: \n\t /usr/bin/python scripts/generate_statistics_plots.py graml \"MiniGrid-Walls-13x13-v0/fragmented_partial_obs/inference_same_length/learn_same_length\" \n Example 2:\n\t /usr/bin/python scripts/generate_statistics_plots.py graql \"MiniGrid-Walls-13x13-v0/fragmented_partial_obs\""
	confs = sys.argv[2].split("/")
	analyze_and_produce_plots(sys.argv[1], confs)