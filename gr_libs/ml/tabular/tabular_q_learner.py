# Don't import stuff from metrics! it's a higher level module.
import os.path
import pickle
import random
from types import MethodType

import dill
from gymnasium import register
import numpy as np

from tqdm import tqdm
from typing import Any
from random import Random
from typing import List, Iterable
from gymnasium.error import InvalidAction
from gr_libs.environment.environment import QLEARNING, MinigridProperty
from gr_libs.ml.tabular import TabularState
from gr_libs.ml.tabular.tabular_rl_agent import TabularRLAgent
from gr_libs.ml.utils import get_agent_model_dir, random_subset_with_order, softmax


class TabularQLearner(TabularRLAgent):
    """
    A simple Tabular Q-Learning agent.
    """

    MODEL_FILE_NAME = r"tabular_model.txt"
    CONF_FILE = r"conf.pkl"

    def __init__(self,
                 domain_name: str,
                 problem_name: str,
                 algorithm: str,
                 num_timesteps: int,
                 decaying_eps: bool = True,
                 eps: float = 1.0,
                 alpha: float = 0.5,
                 decay: float = 0.000002,
                 gamma: float = 0.9,
                 rand: Random = Random(),
                 learning_rate: float = 0.001,
                 check_partial_goals: bool = True,
                 valid_only: bool = False
                 ):
        super().__init__(
            domain_name=domain_name,
            problem_name=problem_name,
            episodes=num_timesteps,
            decaying_eps=decaying_eps,
            eps=eps,
            alpha=alpha,
            decay=decay,
            gamma=gamma,
            rand=rand,
            learning_rate=learning_rate
        )
        assert algorithm == QLEARNING, f"algorithm {algorithm} is not supported by {self.__class__.__name__}"
        self.valid_only = valid_only
        self.check_partial_goals = check_partial_goals
        self.goal_literals_achieved = set()
        self.model_directory = get_agent_model_dir(domain_name=domain_name, model_name=problem_name, class_name=self.class_name())
        self.model_file_path = os.path.join(self.model_directory, TabularQLearner.MODEL_FILE_NAME)
        self._conf_file = os.path.join(self.model_directory, TabularQLearner.CONF_FILE)

        self._learned_episodes = 0

        if os.path.exists(self.model_file_path):
            print(f"Loading pre-existing model in {self.model_file_path}")
            self.load_q_table(path=self.model_file_path)
        else:
            print(f"Creating new model in {self.model_file_path}")
        if os.path.exists(self._conf_file):
            print(f"Loading pre-existing conf file in {self._conf_file}")
            with open(self._conf_file, "rb") as f:
                conf = dill.load(file=f)
            self._learned_episodes = conf['learned_episodes']

        # hyperparameters
        self.base_eps = eps
        self.patience = 400000
        if self.decaying_eps:
            def epsilon():
                self._c_eps = max((self.episodes - self.step) / self.episodes, 0.01)
                return self._c_eps

            self.eps = epsilon
        else:
            self.eps = lambda: eps
        self.decaying_eps = decaying_eps
        self.alpha = alpha
        self.last_state = None
        self.last_action = None

    def states_in_q(self) -> Iterable:
        """Returns the states stored in the q_values table

        Returns:
            List: The states for which we have a mapping in the q-table
        """
        return self.q_table.keys()

    def policy(self, state: TabularState) -> Any:
        """Returns the greedy deterministic policy for the specified state

        Args:
            state (State): the state for which we want the action

        Raises:
            InvalidAction: Not sure about this one

        Returns:
            Any: The greedy action learned for state
        """
        return self.best_action(state)

    def epsilon_greedy_policy(self, state: TabularState) -> Any:
        eps = self.eps()
        if self._random.random() <= eps:
            action = self._random.randint(0, self.number_of_actions - 1)
        else:
            action = self.policy(state)
        return action

    def softmax_policy(self, state: TabularState) -> np.array:
        """Returns a softmax policy over the q-value returns stored in the q-table

        Args:
            state (State): the state for which we want a softmax policy

        Returns:
            np.array: probability of taking each action in self.actions given a state
        """
        if str(state) not in self.q_table:
            self.add_new_state(state)
            # If we query a state we have not visited, return a uniform distribution
            # return softmax([0]*self.actions)
        return softmax(self.q_table[str(state)])

    def save_q_table(self, path: str):
        # sadly, this does not work, because the state we are using
        # is a frozenset of literals, which are not serializable.
        # a way to fix this is to use array states built using
        # common_functions.build_state

        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, path: str):
        with open(path, 'rb') as f:
            table = pickle.load(f)
        self.q_table = table

    def add_new_state(self, state: TabularState):
        self.q_table[str(state)] = [0.] * self.number_of_actions

    def get_all_q_values(self, state: TabularState) -> List[float]:
        if str(state) in self.q_table:
            return self.q_table[str(state)]
        else:
            return [0.] * self.number_of_actions

    def best_action(self, state: TabularState) -> float:
        if str(state) not in self.q_table:
            self.add_new_state(state)
        return np.argmax(self.q_table[str(state)])

    def get_max_q(self, state) -> float:
        if str(state) not in self.q_table:
            self.add_new_state(state)
        return np.max(self.q_table[str(state)])

    def set_q_value(self, state: TabularState, action: Any, q_value: float):
        if str(state) not in self.q_table:
            self.add_new_state(state)
        self.q_table[str(state)][action] = q_value

    def get_q_value(self, state: TabularState, action: Any) -> float:
        if str(state) not in self.q_table:
            self.add_new_state(state)
        return self.q_table[str(state)][action]

    def agent_start(self, state: TabularState) -> int:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        self.last_state = state
        self.last_action = self.policy(state)
        return self.last_action

    def agent_step(self, reward: float, state: TabularState) -> int:
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        max_q = self.get_max_q(state)
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = self.gamma * max_q - old_q
        new_q = old_q + self.alpha * (reward + td_error)

        self.set_q_value(self.last_state, self.last_action, new_q)
        # action = self.best_action(state)
        action = self.epsilon_greedy_policy(state)
        self.last_state = state
        self.last_action = action
        return action

    def agent_end(self, reward: float) -> Any:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = - old_q

        new_q = old_q + self.alpha * (reward + td_error)
        self.set_q_value(self.last_state, self.last_action, new_q)

    def learn(self, init_threshold: int = 20):
        tsteps = 2000
        done_times = 0
        patience = 0
        converged_at = None
        max_r = float("-inf")
        print(f"{self._learned_episodes}->{self.episodes}")
        if self._learned_episodes >= self.episodes:
            print("learned episodes is above the requsted episodes")
            return
        print(f'Using {self.__class__.__name__}')
        tq = tqdm(range(self.episodes - self._learned_episodes),
                  postfix=f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self._c_eps:.3f}. MaxR: {max_r}")
        for n in tq:
            self.step = n
            episode_r = 0
            observation, info = self.env.reset()
            tabular_state = TabularState.gen_tabular_state(environment=self.env, observation=observation)
            action = self.agent_start(state=tabular_state)

            self.update_states_counter(observation_str=str(tabular_state))
            done = False
            tstep = 0
            while tstep < tsteps and not done:
                observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated | truncated
                if done:
                    done_times += 1

                # standard q-learning algorithm
                next_tabular_state = TabularState.gen_tabular_state(environment=self.env, observation=observation)
                self.update_states_counter(observation_str=str(next_tabular_state))
                action = self.agent_step(reward, next_tabular_state)
                tstep += 1
                episode_r += reward
            self._learned_episodes = self._learned_episodes + 1
            if done:  # One last update at the terminal state
                self.agent_end(reward)

            if episode_r > max_r:
                max_r = episode_r
                # print("New all time high reward:", episode_r)
                tq.set_postfix_str(
                    f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self._c_eps:.3f}. MaxR: {max_r}")
            if (n + 1) % 100 == 0:
                tq.set_postfix_str(
                    f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self._c_eps:.3f}. MaxR: {max_r}")
            if (n + 1) % 1000 == 0:
                tq.set_postfix_str(
                    f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self._c_eps:.3f}. MaxR: {max_r}")
                if done_times <= 10:
                    patience += 1
                    if patience >= self.patience:
                        print(f"Did not find goal after {n} episodes. Retrying.")
                        raise InvalidAction("Did not learn")
                else:
                    patience = 0
                if done_times == 1000 and converged_at is not None:
                    converged_at = n
                    print(f"***Policy converged to goal at {converged_at}***")
                done_times = 0
            self.goal_literals_achieved.clear()

        print(f"number of unique states found during training:{self.get_number_of_unique_states()}")
        print("finish learning and saving status")
        self.save_models_to_files()

    def exploit(self, number_of_steps=20):
        observation, info = self.env.reset()
        for step_number in range(number_of_steps):
            tabular_state = TabularState.gen_tabular_state(environment=self.env, observation=observation)
            action = self.policy(state=tabular_state)
            observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            if done:
                print(f"reached goal after {step_number + 1} steps!")
                break

    def get_actions_probabilities(self, observation):
        obs, agent_pos = observation
        direction = obs['direction']

        x, y = agent_pos
        tabular_state = TabularState(agent_x_position=x, agent_y_position=y, agent_direction=direction)
        return softmax(self.get_all_q_values(tabular_state))

    def get_q_of_specific_cell(self, cell_key):
        cell_q_table = {}
        for i in range(4):
            key = cell_key + ':' + str(i)
            if key in self.q_table:
                cell_q_table[key] = self.q_table[key]
        return cell_q_table

    def get_all_cells(self):
        cells = set()
        for key in self.q_table.keys():
            cell = key.split(':')[0]
            cells.add(cell)
        return list(cells)


    def _save_conf_file(self):
        conf = {
            'learned_episodes': self._learned_episodes,
            'states_counter': self.states_counter
        }
        with open(self._conf_file, "wb") as f:
            dill.dump(conf, f)

    def save_models_to_files(self):
        self.save_q_table(path=self.model_file_path)
        self._save_conf_file()
        
    def simplify_observation(self, observation):
        return [(obs['direction'], agent_pos_x, agent_pos_y, action) for ((obs, (agent_pos_x, agent_pos_y)), action) in observation] # list of tuples, each tuple the sample
        
    def generate_observation(self, action_selection_method: MethodType, random_optimalism, save_fig = False, fig_path: str=None, env_prop=None):
        """
        Generate a single observation given a list of agents

        Args:
            agents (list): A list of agents from which to select one randomly.
            action_selection_method : a MethodType, to generate the observation stochastically, greedily, or softmax.

        Returns:
            list: A list of state-action pairs representing the generated observation.

        Notes:
            The function randomly selects an agent from the given list and generates a sequence of state-action pairs 
            based on the Q-table of the selected agent. The action selection is stochastic, where each action is 
            selected based on the probability distribution defined by the Q-values in the Q-table.

            The generated sequence terminates when a maximum number of steps is reached or when the environment 
            episode terminates.
        """
        if save_fig == False:
            assert fig_path == None, "You can't specify a vid path when you don't even save the figure."
        else:
            assert fig_path != None, "You must specify a vid path when you save the figure."
        obs, _ = self.env.reset()
        MAX_STEPS = 32
        done = False
        steps = []
        for step_index in range(MAX_STEPS):
            x, y = self.env.unwrapped.agent_pos
            str_state = "({},{}):{}".format(x, y, obs['direction'])
            relevant_actions_idx = 3
            action_probs = self.q_table[str_state][:relevant_actions_idx] / np.sum(self.q_table[str_state][:relevant_actions_idx])  # Normalize probabilities
            if step_index == 0 and random_optimalism:
                # print("in 1st step in generating plan and got random optimalism.")
                std_dev = np.std(action_probs)
                # uniques_sorted = np.unique(action_probs)
                num_of_stds = abs(action_probs[0] - action_probs[2]) / std_dev
                if num_of_stds < 2.1:
                    # sorted_indices = np.argsort(action_probs)
                    # action = np.random.choice([sorted_indices[-1], sorted_indices[-2]])
                    action = np.random.choice([0, 2])
                    if action == 0:
                        steps.append(((obs, self.env.unwrapped.agent_pos), action))
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        assert reward >= 0
                        action = 2
                        step_index += 1
                else: action = action_selection_method(action_probs)
            else:
                action = action_selection_method(action_probs)
            steps.append(((obs, self.env.unwrapped.agent_pos), action))
            obs, reward, terminated, truncated, info = self.env.step(action)
            assert reward >= 0
            done = terminated | truncated
            if done:
                break

        #assert len(steps) >= 2
        if save_fig:
            sequence = [pos for ((state, pos), action) in steps]
            #print(f"sequence to {self.problem_name} is:\n\t{steps}\ngenerating image at {img_path}.")
            print(f"generating sequence image at {fig_path}.")
            env_prop.create_sequence_image(sequence, fig_path, self.problem_name) # TODO change that assumption, cannot assume this is minigrid env

        return steps

    def generate_partial_observation(self, action_selection_method: MethodType, percentage: float, save_fig = False, is_consecutive = True, random_optimalism=True, fig_path=None):
        """
        Generate a single observation given a list of agents

        Args:
            agents (list): A list of agents from which to select one randomly.
            action_selection_method : a MethodType, to generate the observation stochastically, greedily, or softmax.

        Returns:
            list: A list of state-action pairs representing the generated observation.

        Notes:
            The function randomly selects an agent from the given list and generates a sequence of state-action pairs 
            based on the Q-table of the selected agent. The action selection is stochastic, where each action is 
            selected based on the probability distribution defined by the Q-values in the Q-table.

            The generated sequence terminates when a maximum number of steps is reached or when the environment 
            episode terminates.
        """

        steps = self.generate_observation(action_selection_method=action_selection_method, random_optimalism=random_optimalism, save_fig=save_fig, fig_path=fig_path) # steps are a full observation
        result = random_subset_with_order(steps, (int)(percentage * len(steps)), is_consecutive)
        if percentage >= 0.8:
            assert len(result) > 2
        return result
    
if __name__ == "__main__":
    from gr_libs.metrics.metrics import greedy_selection
    import gr_envs # to register everything
    agent = TabularQLearner(domain_name="minigrid", problem_name="MiniGrid-LavaCrossingS9N2-DynamicGoal-1x7-v0")
    agent.generate_observation(greedy_selection, True, True)
    
    # python experiments.py --recognizer graml --domain point_maze --task L5 --partial_obs_type continuing --point_maze_env obstacles --collect_stats --inference_same_seq_len
        