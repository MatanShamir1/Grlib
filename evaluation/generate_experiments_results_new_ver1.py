import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import dill

from gr_libs.ml.utils.storage import get_experiment_results_path, set_global_storage_configs
from scripts.generate_task_specific_statistics_plots import get_figures_dir_path

if __name__ == "__main__":

	fragmented_accuracies = {
		'graml': {
			'panda': {'gd_agent': {
									'0.3': [], # every list here should have number of tasks accuracies in it, since we done experiments for L111-L555. remember each accuracy is an average of #goals different tasks.
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 },
					  'gc_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 }},
			'minigrid': {'obstacles': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 },
						 'lava_crossing': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 }},
			'point_maze': {'obstacles': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 },
					  	   'four_rooms': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 }},
			'parking': {'gd_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 },
						'gc_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 }},
		},
		'graql': {
			'panda': {'gd_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 },
					  'gc_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 }},
			'minigrid': {'obstacles': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 },
						 'lava_crossing': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 }},
			'point_maze': {'obstacles': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 },
					  	   'four_rooms': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 }},
			'parking': {'gd_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 },
						'gc_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 }},
		}
	}

	continuing_accuracies = copy.deepcopy(fragmented_accuracies)
 
	#domains = ['panda', 'minigrid', 'point_maze', 'parking']
	domains = ['minigrid', 'point_maze', 'parking']
	tasks = ['L111', 'L222', 'L333', 'L444', 'L555']
	percentages = ['0.3', '0.5', '0.7', '0.9', '1']

	for partial_obs_type, accuracies, is_same_learn in zip(['fragmented', 'continuing'], [fragmented_accuracies, continuing_accuracies], [False, True]):
		for domain in domains:
			for env in accuracies['graml'][domain].keys():
				for task in tasks:
					set_global_storage_configs(recognizer_str='graml', is_fragmented=partial_obs_type,
											is_inference_same_length_sequences=True, is_learn_same_length_sequences=is_same_learn)
					graml_res_file_path = f'{get_experiment_results_path(domain, env, task)}.pkl'
					set_global_storage_configs(recognizer_str='graql', is_fragmented=partial_obs_type)
					graql_res_file_path = f'{get_experiment_results_path(domain, env, task)}.pkl'
					if os.path.exists(graml_res_file_path):
						with open(graml_res_file_path, 'rb') as results_file:
							results = dill.load(results_file)
							for percentage in accuracies['graml'][domain][env].keys():
								accuracies['graml'][domain][env][percentage].append(results[percentage]['accuracy'])
					else:
						assert(False, f"no file for {graml_res_file_path}")
					if os.path.exists(graql_res_file_path):
						with open(graql_res_file_path, 'rb') as results_file:
							results = dill.load(results_file)
							for percentage in accuracies['graml'][domain][env].keys():
								accuracies['graql'][domain][env][percentage].append(results[percentage]['accuracy'])
					else:
						assert(False, f"no file for {graql_res_file_path}")

	plot_styles = {
		('graml', 'fragmented', 0.3): 'g--o',  # Green dashed line with circle markers
		('graml', 'fragmented', 0.5): 'g--s',  # Green dashed line with square markers
		('graml', 'fragmented', 0.7): 'g--^',  # Green dashed line with triangle-up markers
		('graml', 'fragmented', 0.9): 'g--d',  # Green dashed line with diamond markers
		('graml', 'fragmented', 1.0): 'g--*',  # Green dashed line with star markers
		
		('graml', 'continuing', 0.3): 'g-o',   # Green solid line with circle markers
		('graml', 'continuing', 0.5): 'g-s',   # Green solid line with square markers
		('graml', 'continuing', 0.7): 'g-^',   # Green solid line with triangle-up markers
		('graml', 'continuing', 0.9): 'g-d',   # Green solid line with diamond markers
		('graml', 'continuing', 1.0): 'g-*',   # Green solid line with star markers
		
		('graql', 'fragmented', 0.3): 'b--o',  # Blue dashed line with circle markers
		('graql', 'fragmented', 0.5): 'b--s',  # Blue dashed line with square markers
		('graql', 'fragmented', 0.7): 'b--^',  # Blue dashed line with triangle-up markers
		('graql', 'fragmented', 0.9): 'b--d',  # Blue dashed line with diamond markers
		('graql', 'fragmented', 1.0): 'b--*',  # Blue dashed line with star markers
		
		('graql', 'continuing', 0.3): 'b-o',   # Blue solid line with circle markers
		('graql', 'continuing', 0.5): 'b-s',   # Blue solid line with square markers
		('graql', 'continuing', 0.7): 'b-^',   # Blue solid line with triangle-up markers
		('graql', 'continuing', 0.9): 'b-d',   # Blue solid line with diamond markers
		('graql', 'continuing', 1.0): 'b-*',   # Blue solid line with star markers
	}

	def average_accuracies(accuracies, domain):
		avg_acc = {algo: {perc: [] for perc in percentages} 
				   for algo in ['graml', 'graql']}
		
		for algo in avg_acc.keys():
			for perc in percentages:
				for env in accuracies[algo][domain].keys():
					env_acc = accuracies[algo][domain][env][perc] # list of 5, averages for L111 to L555.
					if env_acc:
						avg_acc[algo][perc].append(np.array(env_acc))
		
		for algo in avg_acc.keys():
			for perc in percentages:
				if avg_acc[algo][perc]:
					avg_acc[algo][perc] = np.mean(np.array(avg_acc[algo][perc]), axis=0)
		
		return avg_acc

	def plot_domain_accuracies(ax, fragmented_accuracies, continuing_accuracies, domain):
		fragmented_avg_acc = average_accuracies(fragmented_accuracies, domain)
		continuing_avg_acc = average_accuracies(continuing_accuracies, domain)
		
		x_vals = np.arange(1, 6)  # Number of goals
		
		# Create "waves" (shaded regions) for each algorithm
		for algo in ['graml', 'graql']:
			for perc in percentages:
				fragmented_y_vals = np.array(fragmented_avg_acc[algo][perc])
				continuing_y_vals = np.array(continuing_avg_acc[algo][perc])
				
				ax.plot(
					x_vals, fragmented_y_vals, 
					plot_styles[(algo, 'fragmented', float(perc))],  # Use the updated plot_styles dictionary with percentage
					label=f"{algo}, non-consecutive, {perc}"
				)
				ax.plot(
					x_vals, continuing_y_vals, 
					plot_styles[(algo, 'continuing', float(perc))],  # Use the updated plot_styles dictionary with percentage
					label=f"{algo}, consecutive, {perc}"
				)
		
		ax.set_xticks(x_vals)
		ax.set_yticks(np.linspace(0, 1, 6))
		ax.set_ylim([0, 1])
		ax.set_title(f'{domain.capitalize()} Domain', fontsize=16)
		ax.grid(True)

	fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # Increase the figure size for better spacing (width 24, height 6)
	
	# Generate each plot in a subplot, including both fragmented and continuing accuracies
	for i, domain in enumerate(domains):
		plot_domain_accuracies(axes[i], fragmented_accuracies, continuing_accuracies, domain)

	# Set a single x-axis and y-axis label for the entire figure
	fig.text(0.5, 0.04, 'Number of Goals', ha='center', fontsize=20)  # Centered x-axis label
	fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=20)  # Reduced spacing for y-axis label

	# Adjust subplot layout to avoid overlap
	plt.subplots_adjust(left=0.09, right=0.91, top=0.76, bottom=0.24, wspace=0.3)  # More space on top (top=0.82)
	
	# Place the legend above the plots with more space between legend and plots
	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05), fontsize=12)  # Moved above with bbox_to_anchor

	# Save the figure and show it
	plt.savefig('accuracy_plots.png', dpi=300)