import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import inspect
import torch
import dill

from generate_embeddings_dynamic_goals import offline_dynamic_tasks_embeddings
from generate_embeddings_source_tasks import offline_trained_agents_embeddings
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
GRAML_itself = os.path.dirname(currentdir)
GRAML_includer = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, GRAML_includer)
sys.path.insert(0, GRAML_itself)

from GRAML.ml.utils import get_embeddings_result_path
from GRAML.ml.utils import get_embeddings_result_path, get_model_dir, problem_list_to_str_tuple
from GRAML.recognizer.graml_recognizer import GramlRecognizer, minigrid_str_to_goal

def get_tasks_embeddings_dir_path(env_name):
	return GRAML_itself + '/' + get_embeddings_result_path(env_name)

def analyze_and_produce_plots(env_name):
	# DS declaration
	source_tasks_embedding_dicts = {}
	tasks_confidences = {}
	plan_policy_similarity = {}
	base_goals_embeddings_similarity = {}

	embeddings_dir_path = get_tasks_embeddings_dir_path(env_name)
	for embeddings_file_name in os.listdir(embeddings_dir_path):
		with open(os.path.join(embeddings_dir_path, embeddings_file_name), 'rb') as emb_file:
			splitted_name = embeddings_file_name.split('_')
			goal, percentage = splitted_name[0], splitted_name[1]
			source_tasks_embedding_dicts[f"{goal}_{percentage}"] = dill.load(emb_file)

	for goal_percentage, embedding_dict in source_tasks_embedding_dicts.items():
		# check base goal similarity to other base goals
		goal, percentage = goal_percentage.split('_')
		similarities = {dynamic_goal: [] for dynamic_goal in embedding_dict.keys() if 'true' not in dynamic_goal}
		real_goal_embedding = embedding_dict[f"{goal}_true"]
		for dynamic_goal, goal_embedding in embedding_dict.items():
			if 'true' in dynamic_goal: continue
			curr_similarity = torch.exp(-torch.sum(torch.abs(goal_embedding-real_goal_embedding)))
			similarities[dynamic_goal].append(curr_similarity.item())
		# source_tasks_embedding_dicts[goal_percentage].pop(f"{goal}_true")
		if goal not in base_goals_embeddings_similarity.keys(): base_goals_embeddings_similarity[goal] = {}
		base_goals_embeddings_similarity[goal][percentage] = similarities

	
	# Prepare data for plotting
	goals = list(base_goals_embeddings_similarity.keys())
	percentages = sorted(set(percentage for similarities in base_goals_embeddings_similarity.values() for percentage in similarities.keys()))
	num_percentages = len(percentages)

	fig, axes = plt.subplots(nrows=num_percentages, ncols=1, figsize=(10, 6 * num_percentages))

	if num_percentages == 1:
		axes = [axes]

	for i, percentage in enumerate(percentages):
		ax = axes[i]
		dynamic_goals = list(next(iter(base_goals_embeddings_similarity.values()))[percentage].keys())
		num_goals = len(goals)
		num_dynamic_goals = len(dynamic_goals)
		bar_width = 0.8 / num_dynamic_goals
		bar_positions = np.arange(num_goals)

		for j, dynamic_goal in enumerate(dynamic_goals):
			similarities = [base_goals_embeddings_similarity[goal][percentage][dynamic_goal][0] for goal in goals]
			ax.bar(bar_positions + j * bar_width, similarities, bar_width, label=f"Similarity to {dynamic_goal}")

		# Set plot labels and title
		ax.set_xlabel('Goals')
		ax.set_ylabel('Similarity')
		ax.set_title(f'Similarity of Goals to Actual Goal ({percentage}%)')
		ax.set_xticks(bar_positions + bar_width * (num_dynamic_goals - 1) / 2)
		ax.set_xticklabels(goals)
		ax.legend()

	# Save the figure
	fig.savefig(f"{env_name}_goal_similarities_combined.png")

	# Show plot
	plt.tight_layout()
	plt.show()
	
if __name__ == "__main__":
	# checks:
	assert len(sys.argv) == 2, f"Assertion failed: len(sys.argv) is {len(sys.argv)} while it needs to be 3.\n Example: \n\t /usr/bin/python scripts/generate_statistics_plots.py MiniGrid-Walls-13x13-v0"
	assert os.path.exists(get_embeddings_result_path(sys.argv[1])), "embeddings weren't made for this environment, run graml_main.py with this environment first."
	analyze_and_produce_plots(sys.argv[1])