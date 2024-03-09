import csv
import os
import torch
import logging
import sys

from .. import utils
from .other import device


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    return "dataset"


def _get_models_directory_name():
    return "models"


def _get_observations_directory_name():
    return "observations"


def get_observation_file_name(observability_percentage: float):
    return 'obs' + str(observability_percentage) + '.pkl'


def get_env_dir(env_name):
    return os.path.join(get_storage_dir(), env_name)


def get_observations_dir(env_name):
    return os.path.join(get_env_dir(env_name=env_name), _get_observations_directory_name())


def get_model_dir(env_name, model_name, class_name):
    return os.path.join(get_env_dir(env_name=env_name), _get_models_directory_name(), model_name, class_name)


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
