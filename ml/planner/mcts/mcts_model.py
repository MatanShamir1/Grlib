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
		state = self.env.reset()
		self.tree.add_node(Node(action=None, action_space=self.action_space, reward=0, terminal=False))

	def expand(self, node):
		action = node.untried_action()
		state, reward, terminated, truncated, _ = self.env.step(action)
		done = terminated | truncated
		new_node = Node(action=action, action_space=self.action_space, reward=reward, terminal=done)
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
		exploration_term = exploration_constant * sqrt(2 * log(parent.num_visits) / child.num_visits)
		return child.performance + exploration_term

	def best_child(self, node, exploration_constant):
		best_child = self.tree.children(node)[0]
		best_value = self.compute_value(node, best_child, exploration_constant)
		iter_children = iter(self.tree.children(node))
		next(iter_children)
		for child in iter_children:
			value = self.compute_value(node, child, exploration_constant)
			if value > best_value:
				best_child = child
				best_value = value
		return best_child

	# only changes the environment to make sure the actions which are already a part of the plan have been executed.
	def execute_partial_plan(self, plan):
		for action in plan:
			state, reward, terminated, truncated, _ = self.env.step(action)
			done = terminated
			if done: return False
		return True

	# finds the ultimate path from the root node to a terminal state (the one that maximized rewards)
	def tree_policy(self):
		node = self.tree.root
		depth = 0
		while not node.terminal:
			if self.tree.is_expandable(node):
				# expansion
				return self.expand(node), depth
			else:
				# selection
				node = self.best_child(node, exploration_constant=1.0/sqrt(2.0))
				# important to simulate the env to get to some state, as the nodes don't hold this information.	
				state, reward, terminated, truncated, _ = self.env.step(node.action)
				# due to stochastity, nodes could sometimes be terminal and sometimes they aren't. important to update it.
				depth += 1
				done = terminated # no way it will be truncated if it is in the tree
				# we only need this node to be terminal in some possible execution and we know the sequence of actions it represents could lead to a goal
				node.terminal |= done	
		return node, depth

	# receives a final state node and updates the rewards of all the nodes on the path to the root
	def backpropagation(self, node, value):
		while node != self.tree.parent(self.tree.root):
			node.num_visits += 1
			node.total_simulation_reward += value
			node.performance = node.total_simulation_reward/node.num_visits
			node = self.tree.parent(node)

	# def forward(self):
	#     self._forward(self.tree.root)

	# def _forward(self,node):
	#     best_child = self.best_child(node, exploration_constant=0)

	#     print("****** {} ******".format(best_child.state))

	#     for child in self.tree.children(best_child):
	#         print("{}: {:0.4f}".format(child.state, child.performance))

	#     if len(self.tree.children(best_child)) > 0:
	#         self._forward(best_child)
	
	def generate_policy_sequence(self):
		self.env.reset()
		trace = []
		node = self.tree.root
		while len(self.tree.children(node)) > 0:
			next = self.best_child(node, exploration_constant=0)
			state, reward, terminated, truncated, _ = self.env.step(next.action) # take the action that lead to the resulting node.
			trace.append(((state, self.env.unwrapped.agent_pos), next.action))
			node = next
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
	steps = 1000
	print(f"No tree found. Executing MCTS, starting with {steps} rollouts for each action.")
	random.seed(2)
	env = gym.make(id=problem_name)
	tree = Tree()
	monteCarloTreeSearch = MonteCarloTreeSearch(env=env, tree=tree)
	original_root = tree.root
	plan = [] # a sequence of actions
	while not tree.root.terminal: # we iterate until the root is a terminal state, meaning the game is over.
		max_reward = 0
		depth = 0
		iteration = 0
		steps = int(steps*0.9)
		print(f"Executing {steps} rollouts for each action now.")
		tq = tqdm(range(steps), postfix=f"Iteration: {iteration}, Number of steps: {len(plan)}. Selection depth: {depth}. Maximum reward: {max_reward}")
		for n in tq:
			iteration = n
			env.reset()
			# when executing the partial plan, it's possible the environment finished due to the stochasticity. the execution would return false if that happend.
			if not monteCarloTreeSearch.execute_partial_plan(plan):
				# false return value from partial plan execution means the plan is finished. we can mark our root as terminal and exit, happy with our plan.
				tree.root.terminal = True
				return save_model_and_generate_policy(tree=tree, original_root=original_root, model_file_path=model_file_path, monteCarloTreeSearch=monteCarloTreeSearch)	
			node, depth = monteCarloTreeSearch.tree_policy() # find a path to a new unvisited node (unique sequence of actions) by utilizing explorative policy or choosing unvisited children recursively
			reward = monteCarloTreeSearch.simulation(node) # proceed from that node randomly and collect the final reward expected from it (heuristic)
			if reward > max_reward:
				max_reward = reward
			monteCarloTreeSearch.backpropagation(node, reward)  # update the performances of nodes along the way until the root
			tq.set_postfix_str(f"Iteration: {iteration}, Number of steps: {len(plan)}. Selection depth: {depth}. Maximum reward: {max_reward}")
		# update the root and start from it next time.
		tree.root = monteCarloTreeSearch.best_child(node=tree.root, exploration_constant=0)
		print(f"Executed action {tree.root.action}")
		plan.append(tree.root.action)
	return save_model_and_generate_policy(tree=tree, original_root=original_root, model_file_path=model_file_path, monteCarloTreeSearch=monteCarloTreeSearch)	
	
	
if __name__ == "__main__":
	# register(
	# 	id="MiniGrid-DynamicGoalEmpty-8x8-3x6-v0",
	# 	entry_point="minigrid.envs:DynamicGoalEmpty",
	# 	kwargs={"size": 8, "agent_start_pos" : (1, 1), "goal_pos": (3,6) },
	# )
	# plan("MiniGrid-DynamicGoalEmpty-8x8-3x6-v0")
	pass