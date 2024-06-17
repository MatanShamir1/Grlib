# import dill
# import torch
# import os
# import sys
# import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# GRAML_itself = os.path.dirname(currentdir)
# GRAML_includer = os.path.dirname(os.path.dirname(currentdir))
# sys.path.insert(0, GRAML_includer)
# sys.path.insert(0, GRAML_itself)
# from GRAML.ml.utils import get_embeddings_result_path

# def get_task_path(task_num, env_name):
#     return GRAML_itself + '/' + get_embeddings_result_path(env_name) + f'/embeddings_dict{task_num}.pkl'

# def main(env_name):
# 	task_num = 0
# 	for goal in ['(6,1)', '(11,3)', '(11,5)', '(11,8)', '(1,7)', '(5,9)']:
# 			for percentage in [0.3, 0.5, 0.7, 0.9, 1]:
# 				with open(get_task_path(task_num, env_name), 'rb') as emb_file:
# 					embeddings_dict = dill.load(emb_file)
# 					goal_embedding = embeddings_dict['actual_goal']
# 					embeddings_dict.pop('actual_goal')
# 					print(embeddings_dict)
# 					for (goal, embedding) in embeddings_dict.items():
# 						curr_similarity = torch.exp(-torch.sum(torch.abs(embedding-goal_embedding)))
# 						print (f'goal: {goal}, similarity: {curr_similarity}')

# if __name__ == "__main__":
# 	if len(sys.argv) != 2:
# 		print("Error: please provide env_name")
# 		exit(1)
# 	if not os.path.exists(get_embeddings_result_path(sys.argv[1])):
# 		print("embeddings weren't made for this environment, run graml_main.py with this environment first.")
# 	main(sys.argv[1])

import dill
import torch
import os
import sys
import inspect
import matplotlib.pyplot as plt
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
GRAML_itself = os.path.dirname(currentdir)
GRAML_includer = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, GRAML_includer)
sys.path.insert(0, GRAML_itself)
from GRAML.ml.utils import get_embeddings_result_path

def get_task_path(task_num, env_name):
    return GRAML_itself + '/' + get_embeddings_result_path(env_name) + f'/embeddings_dict{task_num}.pkl'

def main(env_name):
    task_num = 0
    similarities = {}
    goals = []
    percentages = [0.3, 0.5, 0.7, 0.9, 1]

    for goal in ['(6,1)', '(11,3)', '(11,5)', '(11,8)', '(1,7)', '(5,9)']:
        goals.append(goal)
        similarities[goal] = []

        for percentage in percentages:
            with open(get_task_path(task_num, env_name), 'rb') as emb_file:
                embeddings_dict = dill.load(emb_file)
                goal_embedding = embeddings_dict['actual_goal']
                embeddings_dict.pop('actual_goal')

                for (goal_str, embedding) in embeddings_dict.items():
                    curr_similarity = torch.exp(-torch.sum(torch.abs(embedding-goal_embedding)))
                    similarities[goal].append(curr_similarity.item())

    # Generate the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.15
    x_pos = np.arange(len(goals))

    for i, percentage in enumerate(percentages):
        sim_values = [similarities[goal][i] for goal in goals]
        ax.bar(x_pos + i * bar_width, sim_values, bar_width, label=f'{percentage*100}%')

    ax.set_xlabel('Goals')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity of Goals to Actual Goal')
    ax.set_xticks(x_pos + bar_width * 2)
    ax.set_xticklabels(goals, rotation=90)
    ax.legend()
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('similarity_plot.png')

    # Generate the Markdown report
    with open('report.md', 'w') as report_file:
        report_file.write('# Similarity Report\n\n')
        report_file.write('This report presents the similarities of different goals to the actual goal for varying percentages.\n\n')
        report_file.write('## Similarity Plot\n\n')
        report_file.write('![Similarity Plot](similarity_plot.png)\n\n')
        report_file.write('The plot shows the similarity values for each goal and percentage, where higher values indicate greater similarity to the actual goal.\n')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: please provide env_name")
        exit(1)
    if not os.path.exists(get_embeddings_result_path(sys.argv[1])):
        print("embeddings weren't made for this environment, run graml_main.py with this environment first.")
    main(sys.argv[1])