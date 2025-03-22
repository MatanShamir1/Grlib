# GRLib
GRLib is a python package containing implementations of Goal Recognition (GR) algorithms which use MDPs to represent the decision making process. All agents in those algorithms interact with an environment that's registered in gym API.
## Setup:
If you're on linux, great, If on windows, use git bash for the next commands to work.
1. Find where your python is installed. If you want to find where's your python3.12, you can run:
```sh
py -3.12 -c "import sys; print(sys.executable)"
```
2. Create a new empty venv from that python venv module:
```sh
C:/Users/path/to/Programs/Python/Python312/python.exe -m venv test_env
```
3. Activate the environment:
```sh
source test_env/Scripts/activate
```
4. There's no equivalent to conda env list to check the global virtual environments status, so you can verify the active one via:
```sh
echo $VIRTUAL_ENV
```
5. Install and upgrade basic package management modules:
```sh
/path/to/python.exe -m pip install --upgrade pip setuptools wheel versioneer
```
6. Install the gr_libs package (can add -e for editable mode):
```sh
cd /path/to/clone/of/GoalRecognitionLibs
pip install -e .
```
7. Install gr_lib package (can add -e for editable mode):
```sh
cd /path/to/clone/of/Grlib
pip install -e .
```


<!-- 1. Ensure you have python 3.11 installed.
If you have root permissions, simply use:
```sh
mkdir -p ~/.local/python3.11
dnf install python3.11 --prefix ~/.local/python3.11
echo 'export PATH=$HOME/.local/python3.11/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
Else, use pyenv:
```sh
pyenv install 3.11.0
```
2. Create a new venv or use an existing 3.11 venv, and activate it. To create a new venv:
```sh
~/.pyenv/versions/3.11.0/bin/python -m venv graml_env
./Python-3.11.0/graml_env/bin/activate
```
If you're not a sudo, and you have problems with building python getting such warnings:
```sh
WARNING: The Python ctypes extension was not compiled. Missing the libffi lib?
```
That means you don't have the necesarry libraries for building python, and you probably can't change that since you're not a sudoer.
An alternative solution can be using a conda env:
```sh
conda create -n graml_env python=3.11
conda activate graml_env
```
3. Install GoalRecognitionLibs to get all needed dependencies:
```sh
git clone [GoalRecognitionLibs address]
cd GoalRecognitionLibs
pip install -e . # using the conda's pip of course
``` -->

### Issues & Problems ###
If you're not a sudo, and you have problems with building python getting such warnings:
```sh
WARNING: The Python ctypes extension was not compiled. Missing the libffi lib?
```
That means you don't have the necesarry libraries for building python.

### How to use Grlib ###
Now that you've installed the package, you have additional custom gym environments and you can start creating an ODGR scenario with the algorithm you wish to test.
The tutorial at tutorials/tutorial.py follows a simple ODGR scnenario. We guide through the initialization and deployment process following an example where GRAML is expected to adapt to new emerging goals in the point_maze gym environment.

#### Method 1: write your own script
1. create the recognizer: we need to state the base problems on which the recognizer train.
we also need the env_name for the sake of storing the trained models.
Other notable parameters include the parameters for the training of the model: For example, Graml's LSTM needs to accept input sizes the size of the concatenation of the state space with the action space.

```python
recognizer = Graml(
    env_name="point_maze", # TODO change to macros which are importable from some info or env module of enums.
    problems=[("PointMaze-FourRoomsEnvDense-11x11-Goal-9x1"),
              ("PointMaze-FourRoomsEnv-11x11-Goal-9x9"), # this one doesn't work with dense rewards because of encountering local minima
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-1x9"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x3"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x4"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-8x2"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-3x7"),
              ("PointMaze-FourRoomsEnvDense-11x11-Goal-2x8")],
    task_str_to_goal=maze_str_to_goal,
    method=DeepRLAgent,
    collect_statistics=False,
    train_configs=[(SAC, 200000) for i in range(8)],
    partial_obs_type="fragmented",
    batch_size=32,
    input_size=6,
    hidden_size=8,
    num_samples=20000,
    problem_list_to_str_tuple=lambda problems: "_".join([f"[{s.split('-')[-1]}]" for s in problems]),
    is_learn_same_length_sequences=False,
    goals_adaptation_sequence_generation_method=AGENT_BASED # take expert samples in goals adaptation phase
)
```

2. The domain learning phase: In GRAML's case, the recognizer generates a dataset by training agents towards the base goals and trains an metric model combined of an LSTM on traces generated by those agents.

```python
recognizer.domain_learning_phase()
```
3. The goals adaptation phase: The recognizer receives new goals, along with configurations to the training of those agents - since the sequence generation method in this case is from an expert.
```python
recognizer.goals_adaptation_phase(
    dynamic_goals_problems = ["PointMaze-FourRoomsEnvDense-11x11-Goal-4x4",
                              "PointMaze-FourRoomsEnvDense-11x11-Goal-7x3",
                              "PointMaze-FourRoomsEnvDense-11x11-Goal-3x7"],
    dynamic_train_configs=[(SAC, 200000) for i in range(3)] # for expert sequence generation
)
```
4. Inference phase - this snippet generates a partial sequence by an agent trained towards one of the goals of the inference phase. Note how the trace is generated using a different agent from the recognizer's inner agents or expert (TD3 rather than SAC), and with noise added to every action to simulate suboptimal behavior.
```python
actor = DeepRLAgent(env_name="point_maze", problem_name="PointMaze-FourRoomsEnvDense-11x11-Goal-4x4", algorithm=TD3, num_timesteps=200000)
actor.learn()
full_sequence = actor.generate_observation(
    action_selection_method=stochastic_amplified_selection,
    random_optimalism=True, # the noise that's added to the actions
)
partial_sequence = random_subset_with_order(full_sequence, (int)(0.5 * len(full_sequence)), is_fragmented="fragmented")
closest_goal = recognizer.inference_phase(partial_sequence, maze_str_to_goal("PointMaze-FourRoomsEnvDense-11x11-Goal-4x4"), 0.5)
```
5. check the result returned by GRAML and print whether it was right or not.
```python
print(f"closest_goal returned by GRAML: {closest_goal}\nactual goal actor aimed towards: (4, 4)")
```

#### Method 2: use a configuration file
The configuraiton file consts.py holds configurations of ODGR problems.
You can either use existing ones or add new ones.
Note that using the config file, despite being easier on a large scale, some inputs to the ODGR problem are not as flexible as they would be using method 1.
For example, the sequence generation will be performed by trained agents and is non configurable. The sequences will either be completely consecutive or randomly sampled from the trace.
Example for a problem:

You can use odgr_executor.py to execute a single task:
```sh
python odgr_executor.py --recognizer MCTSBasedGraml --domain minigrid --task L1 --minigrid_env MinigridSimple
```


## Supported Algorithms

| **Name**         | **Supervised**      | **RL**          | **Discrete**     | **Continuous** | **Model-Based**  | **Model-Free** | **Actions Only** |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| GRAQL   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :heavy_check_mark: | :x: |
| DRACO   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: |
| GRAML   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: |

## Supported Domains

| **Name**         | **Action**      | **State**          |
| ------------------- | ------------------ | ------------------ |
| Minigrid   | Discrete | Discrete |
|  PointMaze  | Continuous | Continuous |
| Parking   | Continuous | Continuous |
| Panda   | Continuous | Continuous |

### Experiments
Given here is a guide for executing the experiments. There are benchmark domains suggested in the repository, and the 'scripts' directory suggests a series of tools to analyze them. They are defaultly set on the domains used for GRAML and GRAQL analysis during the writing of GRAML paper, but can easily be adjusted for new domains and algorithms.
1. analyze_results_cross_alg_cross_domain.py: this script runs with no arguments. it injects information from get_experiment_results_path (for example: dataset\graml\minigrid\continuing\inference_same_seq_len\learn_diff_seq_len\experiment_results\obstacles\L111\experiment_results.pkl), and produces a plot with 4 figures showing the accuracy trend of algorithms on the domains checked one against the other. Currently GRAML is checked against GRAQL or DRACO but it can easily be adjusted from within the script.
2. generate_task_specific_statistics_plots.py - this script produces, for a specific task execution (results of execution of experiments.py), a summary combined of a figure with sticks with the accuracies and confidence levels of an algorithm on the task on the varying percentages. figures\point_maze\obstacles\graql_point_maze_obstacles_fragmented_stats.png is an example of a path at which the output is dumped. Another product of this script is a confusion matrix with the confidence levels - visualizing the same data, and the output file resides in this path: figures\point_maze\obstacles\graml_point_maze_obstacles_fragmented_inference_same_seq_len_learn_diff_seq_len_goals_conf_mat.png.

### How to add a new environment
1. bla
2. blalba

### How to add a new Learner
