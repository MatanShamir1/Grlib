import random
from ml.sequential.lstm_model import LstmObservations
from ml import utils
import torch
import gym
from metrics.metrics import stochastic_amplified_selection

from ml.tabular.tabular_q_learner import TabularQLearner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env_name = "MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0"
    agent = TabularQLearner("MiniGrid-Walls-13x13-v0", "MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0")
    agent.learn()
    env = agent.env
    obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
    model = LstmObservations(obs_space=obs_space, action_space=env.action_space, is_continuous=True)
    model.load_state_dict(torch.load("dataset/MiniGrid-Walls-13x13-v0/models/[11x1]_[11x11]_[1x11]/GramlRecognizer/lstm_cnn_model_cont.pth", map_location=device))
    obs = agent.generate_partial_observation(stochastic_amplified_selection, percentage=random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1]))
    embedding = model.embed_sequence_cont(obs, preprocess_obss)
    print(embedding)

if __name__ == "__main__":
	main()