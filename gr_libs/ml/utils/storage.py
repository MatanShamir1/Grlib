import os


def create_folders_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_outputs_dir():
    return "outputs"


def get_recognizer_outputs_dir(recognizer: str):
    return os.path.join(get_outputs_dir(), recognizer)


def get_gr_cache_dir():
    # Prefer local directory if it exists (e.g., in GitHub workspace)
    if os.path.exists("gr_cache"):
        return "gr_cache"
    # Fall back to pre-mounted directory (e.g., in Docker container)
    if os.path.exists("/gr_cache"):
        return "/gr_cache"
    # Default to "dataset" even if it doesn't exist (e.g., will be created)
    return "gr_cache"


def get_trained_agents_dir():
    # Prefer local directory if it exists (e.g., in GitHub workspace)
    if os.path.exists("trained_agents"):
        return "trained_agents"
    # Fall back to pre-mounted directory (e.g., in Docker container)
    if os.path.exists("/trained_agents"):
        return "/trained_agents"
    # Default to "dataset" even if it doesn't exist (e.g., will be created)
    return "trained_agents"


def _get_siamese_datasets_directory_name():
    return "siamese_datasets"


def _get_observations_directory_name():
    return "observations"


def get_observation_file_name(observability_percentage: float):
    return "obs" + str(observability_percentage) + ".pkl"


def get_domain_outputs_dir(domain_name, recognizer: str):
    return os.path.join(get_recognizer_outputs_dir(recognizer), domain_name)


def get_env_outputs_dir(domain_name, env_name, recognizer: str):
    return os.path.join(get_domain_outputs_dir(domain_name, recognizer), env_name)


def get_observations_dir(domain_name, env_name, recognizer: str):
    return os.path.join(
        get_env_outputs_dir(
            domain_name=domain_name, env_name=env_name, recognizer=recognizer
        ),
        _get_observations_directory_name(),
    )


def get_agent_model_dir(domain_name, model_name, class_name):
    return os.path.join(
        get_trained_agents_dir(),
        domain_name,
        model_name,
        class_name,
    )


def get_lstm_model_dir(domain_name, env_name, model_name, recognizer: str):
    return os.path.join(
        get_gr_cache_dir(), recognizer, domain_name, env_name, model_name
    )


### GRAML PATHS ###


def get_siamese_dataset_path(domain_name, env_name, model_name, recognizer: str):
    return os.path.join(
        get_lstm_model_dir(domain_name, env_name, model_name, recognizer),
        _get_siamese_datasets_directory_name(),
    )


def get_embeddings_result_path(domain_name, env_name, recognizer: str):
    return os.path.join(
        get_env_outputs_dir(domain_name, env_name=env_name, recognizer=recognizer),
        "goal_embeddings",
    )


def get_and_create(path):
    create_folders_if_necessary(path)
    return path


def get_experiment_results_path(domain, env_name, task, recognizer: str):
    return os.path.join(
        get_env_outputs_dir(domain, env_name=env_name, recognizer=recognizer),
        task,
        "experiment_results",
    )


def get_plans_result_path(domain_name, env_name, recognizer: str):
    return os.path.join(
        get_env_outputs_dir(domain_name, env_name=env_name, recognizer=recognizer),
        "plans",
    )


def get_policy_sequences_result_path(domain_name, env_name, recognizer: str):
    return os.path.join(
        get_env_outputs_dir(domain_name, env_name, recognizer=recognizer),
        "policy_sequences",
    )


### END GRAML PATHS ###

### GRAQL PATHS ###


def get_gr_as_rl_experiment_confidence_path(domain_name, env_name, recognizer: str):
    return os.path.join(
        get_env_outputs_dir(
            domain_name=domain_name, env_name=env_name, recognizer=recognizer
        ),
        "confidence",
    )


### GRAQL PATHS ###
