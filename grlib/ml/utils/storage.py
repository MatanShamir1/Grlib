import csv
import os
import torch
import logging
import sys

from .other import device

def create_folders_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_storage_framework_dir(recognizer: str):
    return os.path.join(get_storage_dir(),recognizer)

def get_storage_dir():
    return "dataset"

def _get_models_directory_name():
    return "models"

def _get_siamese_datasets_directory_name():
    return "siamese_datasets"

def _get_observations_directory_name():
    return "observations"

def get_observation_file_name(observability_percentage: float):
    return 'obs' + str(observability_percentage) + '.pkl'


def get_domain_dir(domain_name, recognizer:str):
    return os.path.join(get_storage_framework_dir(recognizer), domain_name)

def get_env_dir(domain_name, env_name, recognizer:str):
    return os.path.join(get_domain_dir(domain_name, recognizer), env_name)

def get_observations_dir(domain_name, env_name, recognizer:str):
    return os.path.join(get_env_dir(domain_name=domain_name, env_name=env_name, recognizer=recognizer), _get_observations_directory_name())

def get_agent_model_dir(domain_name, model_name, class_name):
    return os.path.join(get_storage_dir(), _get_models_directory_name(), domain_name, model_name, class_name)

def get_lstm_model_dir(domain_name, env_name, model_name, recognizer:str):
    return os.path.join(get_env_dir(domain_name=domain_name, env_name=env_name, recognizer=recognizer), model_name)

def get_models_dir(domain_name, env_name, recognizer:str):
    return os.path.join(get_env_dir(domain_name=domain_name, env_name=env_name, recognizer=recognizer), _get_models_directory_name())

### GRAML PATHS ###

def get_siamese_dataset_path(domain_name, env_name, model_name, recognizer:str):
    return os.path.join(get_lstm_model_dir(domain_name, env_name, model_name, recognizer), _get_siamese_datasets_directory_name())

def get_embeddings_result_path(domain_name, env_name, recognizer:str):
    return os.path.join(get_env_dir(domain_name, env_name=env_name, recognizer=recognizer), "goal_embeddings")

def get_embeddings_result_path(domain_name, env_name, recognizer:str):
    return os.path.join(get_env_dir(domain_name, env_name=env_name, recognizer=recognizer), "goal_embeddings")

def get_and_create(path):
    create_folders_if_necessary(path)
    return path

def get_experiment_results_path(domain, env_name, task, recognizer:str):
    return os.path.join(get_env_dir(domain, env_name=env_name, recognizer=recognizer), "experiment_results", env_name, task, "experiment_results")

def get_plans_result_path(domain_name, env_name, recognizer:str):
    return os.path.join(get_env_dir(domain_name, env_name=env_name, recognizer=recognizer), "plans")

def get_policy_sequences_result_path(domain_name, env_name, recognizer:str):
    return os.path.join(get_env_dir(domain_name, env_name, recognizer=recognizer), "policy_sequences")

### END GRAML PATHS ###
''
### GRAQL PATHS ###

def get_gr_as_rl_experiment_confidence_path(domain_name, env_name, recognizer:str):
    return os.path.join(get_env_dir(domain_name=domain_name, env_name=env_name, recognizer=recognizer), "experiments")

### GRAQL PATHS ###

def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=device)


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
