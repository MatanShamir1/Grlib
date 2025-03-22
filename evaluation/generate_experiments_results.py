import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import dill
from gr_libs.ml.utils.storage import get_experiment_results_path, set_global_storage_configs
from scripts.generate_task_specific_statistics_plots import get_figures_dir_path

def gen_graph(graph_name, x_label_str, tasks, panda_env, minigrid_env, parking_env, maze_env, percentage):

	fragmented_accuracies = {
		'graml': {
			#'panda': [],
			#'minigrid': [],
			#'point_maze': [],
			'parking': []
		},
		'graql': {
			#'panda': [],
			#'minigrid': [],
			#'point_maze': [],
			'parking': []
		}
	}

	continuing_accuracies = copy.deepcopy(fragmented_accuracies)
 
	#domains_envs = [('minigrid', minigrid_env), ('point_maze', maze_env), ('parking', parking_env)]
	domains_envs = [('parking', parking_env)]

	for partial_obs_type, accuracies, is_same_learn in zip(['fragmented', 'continuing'], [fragmented_accuracies, continuing_accuracies], [False, True]):
		for domain, env in domains_envs:
			for task in tasks:
				set_global_storage_configs(recognizer_str='graml', is_fragmented=partial_obs_type,
										is_inference_same_length_sequences=True, is_learn_same_length_sequences=is_same_learn)
				graml_res_file_path = f'{get_experiment_results_path(domain, env, task)}.pkl'
				set_global_storage_configs(recognizer_str='graql', is_fragmented=partial_obs_type)
				graql_res_file_path = f'{get_experiment_results_path(domain, env, task)}.pkl'
				if os.path.exists(graml_res_file_path):
					with open(graml_res_file_path, 'rb') as results_file:
						results = dill.load(results_file)
						accuracies['graml'][domain].append(results[percentage]['accuracy'])
				else:
					assert(False, f"no file for {graml_res_file_path}")
				if os.path.exists(graql_res_file_path):
					with open(graql_res_file_path, 'rb') as results_file:
						results = dill.load(results_file)
						accuracies['graql'][domain].append(results[percentage]['accuracy'])
				else:
					assert(False, f"no file for {graql_res_file_path}")

	def plot_accuracies(accuracies, partial_obs_type):
		plt.figure(figsize=(10, 6))
		colors = plt.cm.get_cmap('tab10', len(accuracies['graml']) * len(accuracies['graml']['parking']))

		# Define different line styles for each algorithm
		line_styles = {'graml': '-', 'graql': '--'}
		x_vals = np.arange(3, 8)
		plt.xticks(x_vals)
		plt.yticks(np.linspace(0, 1, 6))
		plt.ylim([0, 1])
		# Plot each domain-env pair's accuracies with different line styles for each algorithm
		for alg in ['graml', 'graql']:
			for idx, (domain, acc_values) in enumerate(accuracies[alg].items()):
				if acc_values and len(acc_values) > 0:  # Only plot if there are values
					x_values = np.arange(3, len(acc_values) + 3)
					plt.plot(x_values, acc_values, marker='o', linestyle=line_styles[alg],
							color=colors(idx), label=f"{alg}-{domain}-{partial_obs_type}-{percentage}")

		# Set labels, title, and grid
		plt.xlabel(x_label_str)
		plt.ylabel('Accuracy')
		plt.grid(True)

		# Add legend to differentiate between domain-env pairs
		plt.legend()

		# Save the figure
		fig_path = os.path.join(f"{graph_name}_{partial_obs_type}.png")
		plt.savefig(fig_path)
		print(f"Accuracies figure saved at: {fig_path}")

	print(f'fragmented_accuracies: {fragmented_accuracies}')
	plot_accuracies(fragmented_accuracies, 'fragmented')
	print(f'continuing_accuracies: {continuing_accuracies}')
	plot_accuracies(continuing_accuracies, 'continuing')
 
if __name__ == "__main__":
	#gen_graph("increasing_base_goals", "Number of base goals", ['L1', 'L2', 'L3', 'L4', 'L5'], panda_env='gd_agent', minigrid_env='obstacles', parking_env='gd_agent', maze_env='obstacles')
	#gen_graph("increasing_dynamic_goals", "Number of dynamic goals", ['L1', 'L2', 'L3', 'L4', 'L5'], panda_env='gc_agent', minigrid_env='lava_crossing', parking_env='gc_agent', maze_env='four_rooms')
	gen_graph("base_problems", "Number of goals", ['L111', 'L222', 'L333', 'L444', 'L555'], panda_env='gd_agent', minigrid_env='obstacles', parking_env='gc_agent', maze_env='obstacles', percentage='0.7')