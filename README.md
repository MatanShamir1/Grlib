# GRLib
GRLib is a Python package that implements Goal Recognition (GR) algorithms using Markov Decision Processes (MDPs) to model decision-making processes. These implementations adhere to the Gymnasium API. All agents in these algorithms interact with environments registered to the Gym API as part of the initialization process of the `gr_envs` package, on which GRLib depends. More details on `gr_envs` can be found at: [GR Envs Repository](https://github.com/MatanShamir1/GREnvs).

## Setup

**Note:** If you are using Windows, use Git Bash for the following commands. Otherwise, any terminal or shell will work.

`gr_libs` depends on `gr_envs`, which registers a set of Gym environments. Ensure your Python environment is set up with Python >= 3.11.

### Setting Up a Python Environment (if needed)
#### Using Pip
1. **Find Your Python Installation:**  
   To locate your Python 3.12 executable, run:
   ```sh
   py -3.12 -c "import sys; print(sys.executable)"
   ```
2. **Create a New Virtual Environment:**  
   Using the path found above, create a new empty venv:
   ```sh
   C:/Users/path/to/Programs/Python/Python312/python.exe -m venv test_env
   ```
3. **Activate the Virtual Environment:**
   ```sh
   source test_env/Scripts/activate
   ```
4. **Verify the Active Environment:**  
   Since there is no direct equivalent to `conda env list`, you can check your active environment via:
   ```sh
   echo $VIRTUAL_ENV
   ```

#### Using Conda
If you prefer using Conda, follow these steps:

1. **Create a New Conda Environment:**  
   Replace `3.12` with your desired Python version if necessary.
   ```sh
   conda create -n new_env python=3.12
   ```
2. **Activate the Environment:**
   ```sh
   conda activate new_env
   ```
  
  
### Upgrade Basic Package Management Modules:
   Run the following command (replace `/path/to/python.exe` with the actual path):
   ```sh
   /path/to/python.exe -m pip install --upgrade pip setuptools wheel versioneer
   ```
### Install the `GoalRecognitionLibs` Package:
  The extras install the custom environments defined in `gr_envs`.
  (For editable installation, add the `-e` flag by cloning the repo and cd'ing to it https://github.com/MatanShamir1/GRLib.git)
  - **Minigrid Environment:**  
    ```sh
    pip install gr_libs[minigrid]
    ```
  - **Highway Environment (Parking):**  
    ```sh
    pip install gr_libs[highway]
    ```
  - **Maze Environment (Point-Maze):**  
    ```sh
    pip install gr_libs[maze]
    ```
  - **Panda Environment:**  
    ```sh
    pip install gr_libs[panda]
    ```
   (For editable installation, add the `-e` flag.)
   ```sh
   cd /path/to/clone/of/GoalRecognitionLibs
   pip install -e .
   ```

## Issues & Troubleshooting

For any issues or troubleshooting, please refer to the repository's issue tracker.

## Supported Algorithms

| **Algorithm**        | **Supervised** | **Reinforcement Learning** | **Discrete States** | **Continuous States** | **Discrete Actions** | **Continuous Actions** | **Model-Based** | **Model-Free** | **Action-Only** | **Goal Conditioned** | **Fine-Tuning** | **Supported Environments**                |
|---------------------|----------------|---------------------------|---------------------|----------------------|----------------------|-----------------------|------------------|----------------|----------------|---------------------|-----------------|-------------------------------------------|
| Graql               | ❌             | ✅                        | ✅                  | ❌                   | ✅                   | ❌                    | ❌               | ✅             | ❌             | ❌                  | ❌              | Minigrid                                   |
| Draco               | ❌             | ✅                        | ✅                  | ✅                   | ✅                   | ✅                    | ❌               | ✅             | ❌             | ❌                  | ❌              | MinigridSimple, MinigridLava, PointMazeObstacles, PointMazeFourRooms, PandaReach, Parking            |
| GCDraco             | ❌             | ✅                        | ✅                  | ✅                   | ✅                   | ✅                    | ❌               | ✅             | ❌             | ✅                  | ❌              | PointMazeObstacles, PointMazeFourRooms, PandaReach, Parking                       |
| GCAura              | ❌             | ✅                        | ✅                  | ✅                   | ✅                   | ✅                    | ❌               | ✅             | ❌             | ✅                  | ✅              | PointMaze, PandaReach, Parking            |
| ExpertBasedGraml    | ✅             | ✅                        | ✅                  | ✅                   | ✅                   | ✅                    | ❌               | ✅             | ✅             | ❌                  | ❌              | MinigridSimple, MinigridLava, PointMazeObstacles, PointMazeFourRooms, PandaReach, Parking                       |
| GCGraml             | ✅             | ✅                        | ✅                  | ✅                   | ✅                   | ✅                    | ❌               | ✅             | ✅             | ✅                  | ❌              | PointMazeObstacles, PointMazeFourRooms, PandaReach, Parking                       |


## Supported Domains

| **Domain**  | **Action Space** | **State Space** |
|------------|----------------|----------------|
| Minigrid   | Discrete       | Discrete       |
| PointMaze  | Continuous     | Continuous     |
| Parking    | Continuous     | Continuous     |
| Panda      | Continuous     | Continuous     |

## Supported Environments

| **Domain**  | **Environment name** | **GC Adaptable** |
|------------|----------------|----------------|
| Minigrid   | MiniGrid-SimpleCrossingS13N4       | ❌ |
| Minigrid   | MiniGrid-LavaCrossingS9N2       | ❌ |
| PointMaze  | PointMaze-FourRoomsEnvDense-11x11     | ✅ |
| PointMaze  | PointMaze-ObstaclesEnvDense-11x11     | ✅ |
| Parking    | Parking-S-14-PC-     | ✅ |
| Panda      | PandaMyReachDense     | ✅ |

Do note one can create other environments outside the supported environments, but they're not a part of the benchmark.

## Usage Guide

After installing GRLib, you will have access to custom Gym environments, allowing you to set up and execute an Online Dynamic Goal Recognition (ODGR) scenario with the algorithm of your choice.

Tutorials demonstrating basic ODGR scenarios is available in the sub-package `tutorials`. These tutorials walk through the initialization and deployment process, showcasing how different GR algorithms adapt to emerging goals in various Gym environments.

## Working with an initial dataset of trained agents
gr_libs also includes a library of trained agents for the various supported environments within the package.
To get the dataset of trained agents, you can run:
```sh
pip install gdown
python download_dataset.py
```
Alternatively, you can visit the google-drive links where download_dataset.py points to and manually download the zipped folders, and unzip them into the project directory.
An alternative is to use our docker image, which includes the dataset in it.
You can:
1. pull the image:
```sh
docker pull ghcr.io/MatanShamir1/gr_test_base:latest
```
2. run a container:
```sh
docker run -it ghcr.io/MatanShamir1/gr_test_base:latest bash
```
3. don't forget to install the package from within the container, go back to 'Setup' for that.

### Method 1: Writing a Custom Script

1. **Create a recognizer**
   
   Specify the domain name and specific environment for the recognizer, effectively telling it the domain theory - the collection of states and actions in the environment.

   ```python
   import gr_libs.environment # Triggers gym env registration - you must run it!
   recognizer = Graql(
       domain_name="minigrid",
       env_name="MiniGrid-SimpleCrossingS13N4"
   )
   ```

2. **Domain Learning Phase** (For GRAQL)
   
   GRAQL does not accumulate information about the domain or engage in learning activities during this phase.
   Other algorithms don't require any data for the phase and simply use what's provided in their intialization: the domain and environment specifics, excluding the possible goals.

3. **Goal Adaptation Phase**
   
   The recognizer receives new goals and corresponding training configurations. GRAQL trains goal-directed agents and stores their policies for inference.
   
   ```python
   recognizer.goals_adaptation_phase(
       dynamic_goals=[(11,1), (11,11), (1,11)],
       dynamic_train_configs=[(QLEARNING, 100000) for _ in range(3)]  # For expert sequence generation
   )
   ```

4. **Inference Phase**
   
   This phase generates a partial sequence from a trained agent, simulating suboptimal behavior with Gaussian noise.
   
   ```python
   actor = TabularQLearner(
       domain_name="minigrid",
       problem_name="MiniGrid-SimpleCrossingS13N4-DynamicGoal-11x1-v0",
       algorithm=QLEARNING,
       num_timesteps=100000
   )
   actor.learn()
   full_sequence = actor.generate_observation(
       action_selection_method=stochastic_amplified_selection,
       random_optimalism=True  # Adds noise to action values
   )
   partial_sequence = random_subset_with_order(full_sequence, int(0.5 * len(full_sequence)), is_consecutive=False)
   closest_goal = recognizer.inference_phase(partial_sequence, (11,1), 0.5)
   ```

5. **Evaluate the result**
   
   ```python
   print(f"Closest goal returned by Graql: {closest_goal}\nActual goal actor aimed towards: (11, 1)")
   ```

### Method 2: Using a Configuration File

The `consts.py` file contains predefined ODGR problem configurations. You can use existing configurations or define new ones.

To execute an ODGR problem using the configuration file, you specify a recognizer, a domain, a gym environment within that domain and the task:
```sh
python odgr_executor.py --recognizer ExpertBasedGraml --domain minigrid --task L1 --env_name MiniGrid-SimpleCrossingS13N4
```

If you also add the flag:
```sh
 --collect_stats
```
to the cmd, 3 kinds of outputs will be generated from the ODGR problem's execution:
a. Into:
```sh
outputs\\minigrid\MiniGrid-SimpleCrossingS13N4\MiniGrid-SimpleCrossingS13N4\L1\experiment_results
```
a .pkl and a .txt summary in a dictionary format will be generated, including the summary of all ODGR executions, including runtime and overall accuracies for all lengths and types of input sequences.

b. Into:
```sh
outputs\ExpertBasedGraml\minigrid\MiniGrid-SimpleCrossingS13N4\policy_sequences\MiniGrid-SimpleCrossingS13N4-DynamicGoal-1x11-v0_inference_seq/plan_image.png
```
a visulzation of the sequence the agent generated will be dumped, either in a png or an mp4 format, depending on the domain, for debugability.

c. Into:
either:
```sh
outputs\ExpertBasedGraml\minigrid\MiniGrid-SimpleCrossingS13N4\goal_embeddings
```
In Graml algorithms, or:
```sh
outputs\Graql\minigrid\MiniGrid-SimpleCrossingS13N4\confidence
```
In GRAsRL algorithms,
pickled results from which confidence of the results can be obtained, for offline analysis.

For GRAsRL outputs, for every possible goal, the likelihood of it being the true goal from the input sequence, based on the policy distance metric.

For GRAML outputs, the embeddings of the sequences are pickled for every goal-directed sequence. Offline, since, since in the embdding space of GRAML's metric model- sequences towards the same sequences are close and vice versa, one could reproduce the most likely goal by measuring the elementwise vector distance of the embeddings, and retrieve a confidence of it.

## Running Experiments

In light of the previous section, the user should already know how to scale the experiments using odgr_executor, and they should also understand how to use the 3 types of outputs for offline analysis of the algorithms.
gr_libs also provides another scaling method to run odgr_executor on multiple domains and environments, for many ODGR problems, as well as python scripts for analysis of these results, to create plots and statistics over the executions.

### Scaling odgr_executor runs
A part of the contribution of this package is standardizing the evaluations of MDP-based GR frameworks.
consts.py provides a set of ODGR problems on which the framework can be evaluated.
The 'evaluations' sub-package provides scripts to analyze the results of the all_experiments.py execution, done over the ODGR the problems defined at consts.py.

#### Running all_experiments.py

You can now run `all_experiments.py` with your desired combination of domains, environments, tasks, and recognizers directly from the command line, without editing the script:

```sh
python gr_libs/all_experiments.py \
    --domains minigrid parking \
    --envs MiniGrid-SimpleCrossingS13N4 Parking-S-14-PC- \
    --tasks L1 L2 L3 L4 L5 \
    --recognizers ExpertBasedGraml Graql \
    --n 5
```

- `--domains`: List of domains to run experiments on.
- `--envs`: List of environments (must be in the same order as domains).
- `--tasks`: List of tasks (applied to all domain/env pairs).
- `--recognizers`: List of recognizers/algorithms to evaluate.
- `--n`: Number of times to execute each task (default: 5).

This script uses multiprocessing to simultaneously execute many `odgr_executor.py` runs as child processes. It logs failures and successful executions for debugability.

After execution summary files are generated in `outputs/summaries/` for further analysis and plotting.

another execution example:
```sh
python gr_libs/all_experiments.py --domains parking --envs Parking-S-14-PC- --tasks L1 L2 L3 L4 L5 --recognizers GCAura GCGraml GCDraco BGGraml Draco --n 5
```

### Using analysis scripts
The repository provides benchmark domains and scripts for analyzing experimental results. The `evaluation` directory contains tools for processing and visualizing the results from odgr_executor.py and all_experiments.py.
Please follow the README.md file in the 'evaluation' directory for more details.

## For Developers
Developers will need to work slightly different: instead of installing the packages, they need to clone the repos and either install them as editables or add their paths to PYTHONPATH so they will function as packages effectively.
Additional packages to install as a developer:
```sh
pip install pre-commit
pre-commit install
```
