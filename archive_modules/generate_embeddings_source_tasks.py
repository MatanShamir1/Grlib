import dill
import torch
import os
import sys
import inspect
import ast

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
GRAML_itself = os.path.dirname(currentdir)
GRAML_includer = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, GRAML_includer)
sys.path.insert(0, GRAML_itself)
from GRAML.recognizer.graml_recognizer import GramlRecognizer
from GRAML.ml.utils import get_embeddings_result_path, get_model_dir, problem_list_to_str_tuple
import random
from GRAML.ml.sequential.lstm_model import LstmObservations
from GRAML.ml import utils
import gym
from GRAML.metrics.metrics import stochastic_amplified_selection

from ml.tabular.tabular_q_learner import TabularQLearner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_continuous = False

def offline_trained_agents_embeddings(env_name, problem_names, model_file_path):
    
    # prepare agents
	agents = [TabularQLearner(env_name, problem_name) for problem_name in problem_names]
	for agent in agents: agent.learn()
	env = agents[0].env
	obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
    
	# prepare model
	model = LstmObservations(obs_space=obs_space, action_space=env.action_space, is_continuous=False)
	model.load_state_dict(torch.load(model_file_path, map_location=device))
	model.to(device)  # Ensure model is on the right device	
 
	# generate observations
	obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
	obss = [agent.generate_partial_observation(stochastic_amplified_selection, percentage=random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1])) for agent in agents]
	embeddings = [(agent.problem_name, model.embed_sequence(obs)) for obs in obss]
	return embeddings

if __name__ == "__main__":
	assert len(sys.argv) == 3, f"Assertion failed: len(sys.argv) is {len(sys.argv)} while it needs to be 3.\n Example: \n\t /usr/bin/python scripts/generate_embeddings_source_tasks.py MiniGrid-Walls-13x13-v0 dataset/MiniGrid-Walls-13x13-v0/models/[11x1]_[11x11]_[1x11]_[8x1]/GramlRecognizer/base_problems.conf"
	problems_path = sys.argv[2]
	env_name = sys.argv[1]
	with open(problems_path, 'r') as file:
		content = file.read()
		problem_names = ast.literal_eval(content)
		if not isinstance(problem_names, list): raise ValueError("The content is not a valid Python list.")
			
	model_directory = get_model_dir(env_name=env_name, model_name=problem_list_to_str_tuple(problem_names), class_name=GramlRecognizer.__name__)
	model_file_path = os.path.join(model_directory, r'lstm_cnn_model.pth')
	if not os.path.exists(model_file_path):
		raise FileNotFoundError(f"The path {model_file_path} does not exist (perhaps you haven't trained a GRAML model, run graml_main.py)\n Example: \n\t /usr/bin/python scripts/generate_embeddings_source_tasks.py MiniGrid-Walls-13x13-v0 dataset/MiniGrid-Walls-13x13-v0/models/[11x1]_[11x11]_[1x11]_[8x1]/GramlRecognizer/problems_config.txt")
	offline_trained_agents_embeddings(env_name, problem_names, model_file_path)