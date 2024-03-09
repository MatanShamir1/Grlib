import os

def get_observations_path(env_name: str):
    return f"dataset/{env_name}/observations"

def get_observations_paths(path: str):
    return [os.path.join(path, file_name) for file_name in os.listdir(path)]