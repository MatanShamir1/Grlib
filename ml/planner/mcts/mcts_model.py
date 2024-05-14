import os
import random
from math import sqrt, log

from tqdm import tqdm
import pickle

from ml.utils.storage import get_model_dir
from .utils import Node
from .utils import Tree
from gym.envs.registration import register
import gymnasium as gym
import numpy as np

class MonteCarloTreeSearch():

	def __init__(self, env, tree):
		self.env = env
		self.tree = tree
		self.action_space = self.env.action_space.n
		self.action_space = 3 # currently
		state, _ = self.env.reset()
		self.tree.add_node(Node(identifier=hash((None, None, None, tuple(env.unwrapped.agent_pos), state['direction'], 0)), state=state, action=None, action_space=self.action_space, reward=0, terminal=False, pos=env.unwrapped.agent_pos))

	def expand(self, node, depth):
		action = node.untried_action()
		state, reward, terminated, truncated, _ = self.env.step(action)
		done = terminated | truncated
		new_node = Node(identifier=hash((node.identifier, action, tuple(self.env.unwrapped.agent_pos), state['direction'], depth)), state=state, action=action, action_space=self.action_space, reward=reward, terminal=done, pos=self.env.unwrapped.agent_pos)
		self.tree.add_node(new_node, node)
		return new_node

	def expand_selection_stochastic_node(self, node, resulting_identifier, terminated, truncated, reward, action, state):
		# the new node could result in a terminating state.
		done = terminated | truncated
		new_node = Node(resulting_identifier, state=state, action=action, action_space=self.action_space, reward=reward, terminal=done, pos=self.env.unwrapped.agent_pos)
		self.tree.add_node(new_node, node)
		return new_node

	def simulation(self, node):
		if node.terminal:
			return node.reward
		while True:
			action = random.randint(0, self.action_space-1)
			state, reward, terminated, truncated, _ = self.env.step(action)
			done = terminated | truncated # this time there could be truncation unlike in the tree policy.
			if done:
				return reward

	def compute_value(self, parent, child, exploration_constant):
		exploration_term = exploration_constant * sqrt(2 * log(parent.num_visits) / child.num_visits) # consider not multiplying by 2
		return child.performance + exploration_term

	# return the best action from a node. the value of an action is the weighted sum of performance of all children that are associated with this action.
	def best_action(self, node, exploration_constant):
		tried_actions_values = {} # dictionary mapping actions to tuples of (cumulative number of visits of children, sum of (child performance * num of visits for child)) to compute the mean later
		iter_children = iter(self.tree.children(node))
		for child in iter_children:
			value = self.compute_value(node, child, exploration_constant)
			tried_actions_values.setdefault(child.action, [0, 0]) # create if it doesn't exist
			tried_actions_values[child.action][0] += child.num_visits # add the number of child visits
			tried_actions_values[child.action][1] += value * child.num_visits # add the relative performance of this child
		return max(tried_actions_values, key=lambda k: tried_actions_values[k][1] / tried_actions_values[k][0]) # return the key (action) with the highest average value

	# only changes the environment to make sure the actions which are already a part of the plan have been executed.
	def execute_partial_plan(self, plan):
		node = self.tree.root
		depth = 0
		for action in plan:
			depth += 1
			# important to simulate the env to get to some state, as the nodes don't hold this information.	
			state, reward, terminated, truncated, _ = self.env.step(action)
			done = terminated
			if done: return None, False
			resulting_identifier = hash((node.identifier, action, tuple(self.env.unwrapped.agent_pos), state['direction'], depth))
			node = self.tree.nodes[resulting_identifier]
		return node, True

	# finds the ultimate path from the root node to a terminal state (the one that maximized rewards)
	def tree_policy(self, root_depth):
		node = self.tree.root
		depth = root_depth
		while not node.terminal:
			depth += 1
			if self.tree.is_expandable(node):
				# expansion - in case there's an action that never been tried, its value is infinity to encourage exploration of all children of a node.
				return self.expand(node, depth), depth
			else:
				# selection - balance exploration and exploitation, coming down the tree - but note the selection might lead to new nodes because of stochaticity.
				best_action = self.best_action(node, exploration_constant=1.0/sqrt(2.0))
				# important to simulate the env to get to some state, as the nodes don't hold this information.	
				state, reward, terminated, truncated, _ = self.env.step(best_action)
				# due to stochasticity, nodes could sometimes be terminal and sometimes they aren't. important to update it. also, the resulting state
				# could be a state we've never been at due to uncertainty of actions' outcomes.
				# if the resulting state creates a parent-action-child triplet that hasn't been seen before, add to the tree and return it, similar result to 'expand'.
				resulting_identifier = hash((node.identifier, best_action, tuple(self.env.unwrapped.agent_pos), state['direction'], depth))
				if resulting_identifier not in node.children_identifiers:
					return self.expand_selection_stochastic_node(node, resulting_identifier, terminated, truncated, reward, best_action, state)
				# if got here, the resulting state's was already seen. we can proceed with the tree policy down the tree.
				node = self.tree.nodes[resulting_identifier]
		return node, depth

	# receives a final state node and updates the rewards of all the nodes on the path to the root
	def backpropagation(self, node, value):
		while node != self.tree.parent(self.tree.root):
			node.num_visits += 1
			node.total_simulation_reward += value
			node.performance = node.total_simulation_reward/node.num_visits
			node = self.tree.parent(node)
	
	def generate_policy_sequence(self):
		trace = []
		node = self.tree.root
		while len(self.tree.children(node)) > 0:
			best_action = self.best_action(node, exploration_constant=0)
			candidate_children = [child for child in self.tree.children(node) if child.action == best_action]
			node = max(candidate_children, key=lambda node: node.num_visits)
			trace.append(((node.state, node.pos), node.action))
		return trace

def save_model_and_generate_policy(tree, original_root, model_file_path, monteCarloTreeSearch):
	tree.root = original_root
	with open(model_file_path, 'wb') as file:  # Serialize the model
		monteCarloTreeSearch.env = None # pickle cannot serialize lambdas which exist in the env
		pickle.dump(monteCarloTreeSearch, file)
	return monteCarloTreeSearch.generate_policy_sequence()

def plan(env_name, problem_name):
	model_dir = get_model_dir(env_name=env_name, model_name=problem_name, class_name="MCTS")
	model_file_path = os.path.join(model_dir, "mcts_model.pth")
	if os.path.exists(model_file_path):
		print(f"Loading pre-existing mcts planner in {model_file_path}")
		with open(model_file_path, 'rb') as file:  # Load the pre-existing model
			monteCarloTreeSearch = pickle.load(file)
			return monteCarloTreeSearch.generate_policy_sequence()
	if not os.path.exists(model_dir): # if we reached here, the model doesn't exist. make sure its folder exists.
		os.makedirs(model_dir)
	steps = 500
	print(f"No tree found. Executing MCTS, starting with {steps} rollouts for each action.")
	env = gym.make(id=problem_name)
	random.seed(2)	
	tree = Tree()
	monteCarloTreeSearch = MonteCarloTreeSearch(env=env, tree=tree)
	original_root = tree.root
	plan = [] # a sequence of actions
	depth = 0
	while not tree.root.terminal: # we iterate until the root is a terminal state, meaning the game is over.
		max_reward = 0
		iteration = 0
		steps = int(steps*0.9)
		print(f"Executing {steps} rollouts for each action now.")
		tq = tqdm(range(steps), postfix=f"Iteration: {iteration}, Number of steps: {len(plan)}. Selection depth: {depth}. Maximum reward: {max_reward}")		
		for n in tq:
			iteration = n
			env.reset()
			# when executing the partial plan, it's possible the environment finished due to the stochasticity. the execution would return false if that happend.
			depth = len(plan)
			tree.root = original_root # need to return it to the original root before executing the partial plan as it can lead to a different path and the root can change between iterations.
			node, result = monteCarloTreeSearch.execute_partial_plan(plan)
			if not result:
				# false return value from partial plan execution means the plan is finished. we can mark our root as terminal and exit, happy with our plan.
				tree.root.terminal = True
				return save_model_and_generate_policy(tree=tree, original_root=original_root, model_file_path=model_file_path, monteCarloTreeSearch=monteCarloTreeSearch)
			tree.root = node # determine the roote to be the node executed after the plan for this iteration.
			node, depth = monteCarloTreeSearch.tree_policy(root_depth=depth) # find a path to a new unvisited node (unique sequence of actions) by utilizing explorative policy or choosing unvisited children recursively
			# if the node that returned from tree policy is terminal, the reward will be returned from "simulation" function immediately.
			reward = monteCarloTreeSearch.simulation(node) # proceed from that node randomly and collect the final reward expected from it (heuristic)
			if reward > max_reward:
				max_reward = reward
			monteCarloTreeSearch.backpropagation(node, reward)  # update the performances of nodes along the way until the root
			tq.set_postfix_str(f"Iteration: {iteration}, Number of steps: {len(plan)}. Selection depth: {depth}. Maximum reward: {max_reward}")
		# update the root and start from it next time.
		action = monteCarloTreeSearch.best_action(node=tree.root, exploration_constant=0)
		plan.append(action)
		print(f"Executed action {action}")
	return save_model_and_generate_policy(tree=tree, original_root=original_root, model_file_path=model_file_path, monteCarloTreeSearch=monteCarloTreeSearch)	
	
if __name__ == "__main__":
	# register(
	# 	id="MiniGrid-DynamicGoalEmpty-8x8-3x6-v0",
	# 	entry_point="minigrid.envs:DynamicGoalEmpty",
	# 	kwargs={"size": 8, "agent_start_pos" : (1, 1), "goal_pos": (3,6) },
	# )
	# plan("MiniGrid-DynamicGoalEmpty-8x8-3x6-v0")
	pass