import csv
import os
from typing import List
import torch
import logging
import sys

from .. import utils
from .other import device

IS_FRAGMENTED = None
IS_INFERENCE_SAME_LEN_SEQUENCES = None
IS_LEARN_SAME_LEN_SEQUENCES = None
RECOGNIZER_STR = None
GRAQL = "graql"
GRAML = "graml"

def set_global_storage_configs(recognizer_str, is_fragmented, is_inference_same_length_sequences=None, is_learn_same_length_sequences=None):
    global IS_FRAGMENTED, IS_INFERENCE_SAME_LEN_SEQUENCES, IS_LEARN_SAME_LEN_SEQUENCES, RECOGNIZER_STR
    RECOGNIZER_STR = recognizer_str
    IS_FRAGMENTED = is_fragmented
    IS_INFERENCE_SAME_LEN_SEQUENCES = is_inference_same_length_sequences
    IS_LEARN_SAME_LEN_SEQUENCES = is_learn_same_length_sequences

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    global IS_FRAGMENTED, IS_INFERENCE_SAME_LEN_SEQUENCES, IS_LEARN_SAME_LEN_SEQUENCES, RECOGNIZER_STR
    assert RECOGNIZER_STR == "graql" and IS_FRAGMENTED!=None or (RECOGNIZER_STR == "graml" and IS_FRAGMENTED!=None and IS_INFERENCE_SAME_LEN_SEQUENCES!=None and IS_LEARN_SAME_LEN_SEQUENCES!=None), "You must call 'set_global_storage_configs' before using API from 'storage' module."
    return f"dataset/{RECOGNIZER_STR}"


def _get_models_directory_name():
    return "models"

def _get_datasets_directory_name():
    return "siamese_datasets"

def _get_observations_directory_name():
    return "observations"

def get_observation_file_name(observability_percentage: float):
    return 'obs' + str(observability_percentage) + '.pkl'


def get_env_dir(env_name):
    global IS_FRAGMENTED, IS_INFERENCE_SAME_LEN_SEQUENCES, IS_LEARN_SAME_LEN_SEQUENCES, RECOGNIZER_STR
    if RECOGNIZER_STR == GRAML: return os.path.join(get_storage_dir(), env_name, IS_FRAGMENTED, IS_INFERENCE_SAME_LEN_SEQUENCES, IS_LEARN_SAME_LEN_SEQUENCES)
    else: return os.path.join(get_storage_dir(), env_name)

def get_observations_dir(env_name):
    return os.path.join(get_env_dir(env_name=env_name), _get_observations_directory_name())


def get_model_dir(env_name, model_name, class_name):
    return os.path.join(get_env_dir(env_name=env_name), _get_models_directory_name(), model_name, class_name)

def get_models_dir(env_name):
    return os.path.join(get_env_dir(env_name=env_name), _get_models_directory_name())

### GRAML PATHS ###

def get_siamese_dataset_path(env_name, problem_names):
    return os.path.join(get_env_dir(env_name=env_name), _get_datasets_directory_name(), problem_names)

def get_embeddings_result_path(env_name):
    return os.path.join(get_env_dir(env_name), "goal_embeddings")

def get_plans_result_path(env_name):
    return os.path.join(get_env_dir(env_name), "plans")

def get_policy_sequences_result_path(env_name):
    return os.path.join(get_env_dir(env_name), "policy_sequences")

### END GRAML PATHS ###

### GRAQL PATHS ###

def get_graql_experiment_confidence_path(env_name):
    return os.path.join(get_env_dir(env_name), "experiments")

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

def problem_list_to_str_tuple(problems : List[str]):
    return '_'.join([f"[{s.split('-')[-2]}]" for s in problems])
