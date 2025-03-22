import os
import dill
import random
import hashlib
from typing import List

def get_observations_path(env_name: str):
    return f"dataset/{env_name}/observations"

def get_observations_paths(path: str):
    return [os.path.join(path, file_name) for file_name in os.listdir(path)]

def create_partial_observabilities_files(env_name: str, observabilities: List[float]):
    with open(r"dataset/{env_name}/observations/obs1.0.pkl".format(env_name=env_name), "rb") as f:
        step_1_0 = dill.load(f)

    number_of_items_to_randomize = [int(observability * len(step_1_0)) for observability in observabilities]
    obs = []
    for items_to_randomize in number_of_items_to_randomize:
        obs.append(random.sample(step_1_0, items_to_randomize))
    for index, observability in enumerate(observabilities):
        partial_steps = obs[index]
        file_path = r"dataset/{env_name}/observations/obs{obs}.pkl".format(env_name=env_name, obs=observability)
        with open(file_path, "wb+") as f:
            dill.dump(partial_steps, f)
            
def md5(file_path: str):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_md5(file_path_list: List[str]):
    return [(file_path, md5(file_path=file_path)) for file_path in file_path_list]


def print_md5(file_path_list: List[str]):
    md5_of_observations = get_md5(file_path_list=file_path_list)
    for file_name, file_md5 in md5_of_observations:
        print(f"{file_name}:{file_md5}")
    print("")