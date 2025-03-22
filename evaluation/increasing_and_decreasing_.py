import os
import dill
import numpy as np
import matplotlib.pyplot as plt
from gr_libs.ml.utils.storage import get_experiment_results_path, set_global_storage_configs

if __name__ == "__main__":

	# Define the tasks and percentages
	increasing_base_goals = ['L1', 'L2', 'L3', 'L4', 'L5']
	increasing_dynamic_goals = ['L111', 'L222', 'L555', 'L333', 'L444']
	percentages = ['0.3', '0.5', '0.7', '0.9', '1']
	
	# Prepare a dictionary to hold accuracy data
	accuracies = {task: {perc: [] for perc in percentages} for task in increasing_base_goals + increasing_dynamic_goals}

	# Collect data for both sets of goals
	for task in increasing_base_goals + increasing_dynamic_goals:
		set_global_storage_configs(recognizer_str='graml', is_fragmented='fragmented',
								   is_inference_same_length_sequences=True, is_learn_same_length_sequences=False)
		res_file_path = f'{get_experiment_results_path("parking", "gd_agent", task)}.pkl'
		
		if os.path.exists(res_file_path):
			with open(res_file_path, 'rb') as results_file:
				results = dill.load(results_file)
				for percentage in percentages:
					accuracies[task][percentage].append(results[percentage]['accuracy'])
		else:
			print(f"Warning: no file for {res_file_path}")

	# Create the figure with two subplots
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))

	# Bar plot function
	def plot_accuracies(ax, task_set, title, type):
		"""Plot accuracies for a given set of tasks on the provided axis."""
		x_vals = np.arange(len(task_set))  # X-axis positions for the number of goals
		bar_width = 0.15  # Width of each bar
		for i, perc in enumerate(['0.3', '0.5', '1']):
			if perc == '1': y_vals = [max([accuracies[task]['0.5'][0], accuracies[task]['0.7'][0], accuracies[task]['0.9'][0], accuracies[task]['1'][0]]) for task in task_set]  # Get mean accuracies
			else: y_vals = [accuracies[task][perc][0] for task in task_set]  # Get mean accuracies
			if type != 'base': ax.bar(x_vals + i * bar_width, y_vals, width=bar_width, label=f'Percentage {perc}')
			else: ax.bar(x_vals + i * bar_width, y_vals, width=bar_width)
		ax.set_xticks(x_vals + bar_width)  # Center x-ticks
		ax.set_xticklabels([i+3 for i in range(len(task_set))], fontsize=16)  # Set custom x-tick labels
		ax.set_yticks(np.linspace(0, 1, 6))
		ax.set_ylim([0, 1])
		ax.set_title(title, fontsize=20)
		ax.set_xlabel(f'Number of {type} Goals', fontsize=20)
		if type == 'base':
			ax.set_ylabel('Accuracy', fontsize=22)
		ax.legend()

	# Plot for increasing base goals
	plot_accuracies(axes[0], increasing_base_goals, 'Increasing Base Goals', "base")

	# Plot for increasing dynamic goals
	plot_accuracies(axes[1], increasing_dynamic_goals, 'Increasing Active Goals', "active")
	plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
	# Adjust layout and save the figure
	plt.tight_layout()
	plt.savefig('increasing_goals_plot_bars.png', dpi=300)  # Save the figure as a PNG file
	print('Figure saved at: increasing_goals_plot_bars.png')
