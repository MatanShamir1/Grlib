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

## Usage Guide

After installing GRLib, you will have access to custom Gym environments, allowing you to set up and execute an Online Dynamic Goal Recognition (ODGR) scenario with the algorithm of your choice.

Tutorials demonstrating basic ODGR scenarios is available in the sub-package `tutorials`. These tutorials walk through the initialization and deployment process, showcasing how different GR algorithms adapt to emerging goals in various Gym environments.

## Working with an initial dataset of trained agents
gr_libs also includes a library of trained agents for the various supported environments within the package.
To get the dataset of trained agents, you can run:
```sh
python download_dataset.py
```

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

To execute a single task using the configuration file:
```sh
python odgr_executor.py --recognizer MCTSBasedGraml --domain minigrid --task L1 --minigrid_env MinigridSimple
```

## Supported Algorithms

Successors of algorithms that don't differ in their specifics are added in parentheses after the algorithm name. For example, since GC-DRACO and DRACO share the same column values, they're written on one line as DRACO (GC).

| **Algorithm** | **Supervised** | **Reinforcement Learning** | **Discrete States** | **Continuous States** | **Discrete Actions** | **Continuous Actions** | **Model-Based** | **Model-Free** | **Action-Only** |
|--------------|--------------|------------------------|------------------|------------------|--------------|--------------|--------------|--------------|--------------|
| GRAQL       | ❌           | ✅                     | ✅                | ❌                | ✅                | ❌                | ❌           | ✅           | ❌           |
| DRACO (GC)  | ❌           | ✅                     | ✅                | ✅                | ✅                | ✅                | ❌           | ✅           | ❌           |
| GRAML (GC, BG) | ✅        | ✅                     | ✅                | ✅                | ✅                | ✅                | ❌           | ✅           | ✅           |

## Supported Domains

| **Domain**  | **Action Space** | **State Space** |
|------------|----------------|----------------|
| Minigrid   | Discrete       | Discrete       |
| PointMaze  | Continuous     | Continuous     |
| Parking    | Continuous     | Continuous     |
| Panda      | Continuous     | Continuous     |

## Running Experiments

The repository provides benchmark domains and scripts for analyzing experimental results. The `scripts` directory contains tools for processing and visualizing results.

1. **`analyze_results_cross_alg_cross_domain.py`**
   - Runs without arguments.
   - Reads data from `get_experiment_results_path` (e.g., `dataset/graml/minigrid/continuing/.../experiment_results.pkl`).
   - Generates plots comparing algorithm performance across domains.

2. **`generate_task_specific_statistics_plots.py`**
   - Produces task-specific accuracy and confidence plots.
   - Generates a confusion matrix displaying confidence levels.
   - Example output paths:
     - `figures/point_maze/obstacles/graql_point_maze_obstacles_fragmented_stats.png`
     - `figures/point_maze/obstacles/graml_point_maze_obstacles_conf_mat.png`
