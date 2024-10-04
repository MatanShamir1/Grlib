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

import ast
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

def offline_dynamic_tasks_embeddings(env_name, goals):
	task_num = 0
	percentages = [0.3, 0.5, 0.7, 0.9, 1]

	tasks_embeddings_dict = {}

	for goal in goals:
		for percentage in percentages:
			with open(get_task_path(task_num, env_name), 'rb') as emb_file:
				embeddings_dict = dill.load(emb_file)
				embeddings_dict.pop('actual_goal')
				tasks_embeddings_dict[f"{goal}_{percentage}"] = embeddings_dict
    
	return tasks_embeddings_dict

if __name__ == "__main__":
	assert len(sys.argv) == 3, f"Assertion failed: len(sys.argv) is {len(sys.argv)} while it needs to be 3.\n Example: \n\t /usr/bin/python scripts/generate_embeddings_dynamic_goals.py MiniGrid-Walls-13x13-v0 dataset/MiniGrid-Walls-13x13-v0/models/[11x1]_[11x11]_[1x11]_[8x1]/GramlRecognizer/dynamic_goals.conf"
	assert os.path.exists(get_embeddings_result_path(sys.argv[1])), "embeddings weren't made for this environment, run graml_main.py with this environment first."
	problems_path = sys.argv[2]
	with open(problems_path, 'r') as file:
		content = file.read()
		goals_list = ast.literal_eval(content)
		if not isinstance(goals_list, list): raise ValueError("The content is not a valid Python list.")
	offline_dynamic_tasks_embeddings(sys.argv[1], goals_list)