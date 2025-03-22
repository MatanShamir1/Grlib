import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import dill
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from gr_libs.ml.utils.storage import get_experiment_results_path, set_global_storage_configs
from scripts.generate_task_specific_statistics_plots import get_figures_dir_path

def smooth_line(x, y, num_points=300):
	x_smooth = np.linspace(np.min(x), np.max(x), num_points)
	spline = make_interp_spline(x, y, k=3)  # Cubic spline
	y_smooth = spline(x_smooth)
	return x_smooth, y_smooth

if __name__ == "__main__":

	fragmented_accuracies = {
		'graml': {
			'panda': {'gc_agent': {
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
			'parking': {'gc_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
								 },
						'gd_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
						},		
			   			},
		},
		'graql': {
			'panda': {'gc_agent': {
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
			'parking': {'gc_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
						},
						'gd_agent': {
									'0.3': [],
									'0.5': [],
									'0.7': [],
									'0.9': [],	
									'1'	: []
						},	
			   			},
		}
	}

	continuing_accuracies = copy.deepcopy(fragmented_accuracies)
 
	#domains = ['panda', 'minigrid', 'point_maze', 'parking']
	domains = ['parking']
	tasks = ['L555', 'L444', 'L333', 'L222', 'L111']
	percentages = ['0.3', '0.5', '1']

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

	def plot_domain_accuracies(ax, fragmented_accuracies, continuing_accuracies, domain, sigma=1, line_width=1.5):
		fragmented_avg_acc = average_accuracies(fragmented_accuracies, domain)
		continuing_avg_acc = average_accuracies(continuing_accuracies, domain)
		
		x_vals = np.arange(1, 6)  # Number of goals

		# Create "waves" (shaded regions) for each algorithm
		for algo in ['graml', 'graql']:
			fragmented_y_vals_by_percentage = []
			continuing_y_vals_by_percentage = []


			for perc in percentages:
				fragmented_y_vals = np.array(fragmented_avg_acc[algo][perc])
				continuing_y_vals = np.array(continuing_avg_acc[algo][perc])

				# Smooth the trends using Gaussian filtering
				fragmented_y_smoothed = gaussian_filter1d(fragmented_y_vals, sigma=sigma)
				continuing_y_smoothed = gaussian_filter1d(continuing_y_vals, sigma=sigma)

				fragmented_y_vals_by_percentage.append(fragmented_y_smoothed)
				continuing_y_vals_by_percentage.append(continuing_y_smoothed)

				ax.plot(
					x_vals, fragmented_y_smoothed, 
					plot_styles[(algo, 'fragmented', float(perc))], 
					label=f"{algo}, non-consecutive, {perc}",
					linewidth=0.5  # Control line thickness here
				)
				ax.plot(
					x_vals, continuing_y_smoothed, 
					plot_styles[(algo, 'continuing', float(perc))], 
					label=f"{algo}, consecutive, {perc}",
					linewidth=0.5  # Control line thickness here
				)

			# Fill between trends of the same type that differ only by percentage
			# for i in range(len(percentages) - 1):
			# 	ax.fill_between(
			# 		x_vals, fragmented_y_vals_by_percentage[i], fragmented_y_vals_by_percentage[i+1],
			# 		color='green', alpha=0.1  # Adjust the fill color and transparency (for graml)
			# 	)
			# 	ax.fill_between(
			# 		x_vals, continuing_y_vals_by_percentage[i], continuing_y_vals_by_percentage[i+1],
			# 		color='blue', alpha=0.1  # Adjust the fill color and transparency (for graql)
			# 	)
		
		ax.set_xticks(x_vals)
		ax.set_yticks(np.linspace(0, 1, 6))
		ax.set_ylim([0, 1])
		ax.set_title(f'{domain.capitalize()} Domain', fontsize=16)
		ax.grid(True)

	# COMMENT FROM HERE AND UNTIL NEXT FUNCTION FOR BG GC COMPARISON

	# fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # Increase the figure size for better spacing (width 24, height 6)
	
	# # Generate each plot in a subplot, including both fragmented and continuing accuracies
	# for i, domain in enumerate(domains):
	# 	plot_domain_accuracies(axes[i], fragmented_accuracies, continuing_accuracies, domain)

	# # Set a single x-axis and y-axis label for the entire figure
	# fig.text(0.5, 0.04, 'Number of Goals', ha='center', fontsize=20)  # Centered x-axis label
	# fig.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=20)  # Reduced spacing for y-axis label

	# # Adjust subplot layout to avoid overlap
	# plt.subplots_adjust(left=0.09, right=0.91, top=0.79, bottom=0.21, wspace=0.3)  # More space on top (top=0.82)
	
	# # Place the legend above the plots with more space between legend and plots
	# handles, labels = axes[0].get_legend_handles_labels()
	# fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05), fontsize=12)  # Moved above with bbox_to_anchor

	# # Save the figure and show it
	# plt.savefig('accuracy_plots_smooth.png', dpi=300)
 
	# a specific comparison between bg-graml and gc-graml, aka gd_agent and gc_agent "envs":
	def plot_stick_figures(continuing_accuracies, fragmented_accuracies, title):
		fractions = ['0.3', '0.5', '1']

		def get_agent_data(data_dict, domain='graml', agent='gd_agent'):
			return [np.mean(data_dict[domain]['parking'][agent][fraction]) for fraction in fractions]

		# Continuing accuracies for gd_agent and gc_agent
		cont_gd = get_agent_data(continuing_accuracies, domain='graml', agent='gd_agent')
		cont_gc = get_agent_data(continuing_accuracies, domain='graml', agent='gc_agent')

		# Fragmented accuracies for gd_agent and gc_agent
		frag_gd = get_agent_data(fragmented_accuracies, domain='graml', agent='gd_agent')
		frag_gc = get_agent_data(fragmented_accuracies, domain='graml', agent='gc_agent')

		# Debugging: Print values to check if they're non-zero
		print("Continuing GD:", cont_gd)
		print("Continuing GC:", cont_gc)
		print("Fragmented GD:", frag_gd)
		print("Fragmented GC:", frag_gc)

		# Setting up figure
		x = np.arange(len(fractions))  # label locations
		width = 0.35  # width of the bars

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

		# Plot for continuing accuracies
		ax1.bar(x - width / 2, cont_gd, width, label='BG-GRAML')
		ax1.bar(x + width / 2, cont_gc, width, label='GC-GRAML')
		ax1.set_title('Consecutive Sequences', fontsize=20)
		ax1.set_xticks(x)
		ax1.set_xticklabels(fractions, fontsize=16)
		ax1.set_yticks(np.arange(0, 1.1, 0.2))
		ax1.set_yticklabels(np.round(np.arange(0, 1.1, 0.2), 1), fontsize=16)
		ax1.legend(fontsize=20)

		# Plot for fragmented accuracies
		ax2.bar(x - width / 2, frag_gd, width, label='BG-GRAML')
		ax2.bar(x + width / 2, frag_gc, width, label='GC-GRAML')
		ax2.set_title('Non-Consecutive Sequences', fontsize=20)
		ax2.set_xticks(x)
		ax2.set_xticklabels(fractions, fontsize=16)
		ax2.set_yticks(np.arange(0, 1.1, 0.2))
		ax2.set_yticklabels(np.round(np.arange(0, 1.1, 0.2), 1), fontsize=16)
		ax2.set_ylim(0, 1)  # Ensure the y-axis is properly set
		ax2.legend(fontsize=20)
		# Common axis labels
		fig.text(0.5, 0.02, 'Observation Portion', ha='center', va='center', fontsize=24)
		fig.text(0.06, 0.5, 'Accuracy', ha='center', va='center', rotation='vertical', fontsize=24)

		plt.subplots_adjust(top=0.85)
		plt.savefig('gd_vs_gc_parking.png', dpi=300)

	plot_stick_figures(continuing_accuracies, fragmented_accuracies, "GC-GRAML compared with BG-GRAML")