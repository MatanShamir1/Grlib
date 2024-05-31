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
import copy

PROB = 0.8
UNIFORM_PROB = 0.1
newely_expanded = 0
dict_dir_id_to_str = {0:'right', 1:'down', 2:'left', 3:'up'}
# TODO add number of expanded nodes and debug by putting breakpoint on the creation of nodes representing (8,4) and checking if they're invalid or something

# Explanation on hashing and uncertainty in the acto outcome:
# We want to detect circles, while not preventing expected behavior. To achieve it, hasing must include previous state, action, and resulting state.
# Hashing the direction means coming to the same position from different positions gets different id's.
# Example: the agent might have stood at (2,2), picked action 2 (forward), and accidently turned right, resulting at state ((2,2), right). 
#		   later, when the agent stood at (2,1), looked right and walked forward, it got to the same state. We would want to enable that, because
#		   this is the expected behavior, so these nodes must have unique id's.
# The situations where circles will indeed be detected, are only if the outcome was the same for the previous state, consistent with the action - whether it was or wasn't expected.
class MonteCarloTreeSearch():

	def __init__(self, env, tree, goal, use_heuristic=True):
		self.env = env
		self.tree = tree
		self.action_space = self.env.action_space.n
		self.action_space = 3 # currently
		state, _ = self.env.reset()
		self.use_heuristic = use_heuristic
		self.goal = goal
		self.tree.add_node(Node(identifier=hash((None, None, tuple(self.env.unwrapped.agent_pos), state['direction'])), state=state, action=None, action_space=self.action_space, reward=0, terminal=False, pos=env.unwrapped.agent_pos, depth=0))

	# def mark_invalid_children(self, children_identifiers, action):
	# 	for child_id in children_identifiers:
	# 		child = self.tree.nodes[child_id]
	# 		if child.action == action:
	# 			child.invalid = True

	def decide_invalid_path(self, new_node_father, old_node, new_node): # new_node created the circle, old_node got to the configuration first.
		new_visits, old_visits = [1,1], [0,0] # stochasticity couldn't result a cycle directly, because it involves a different action. we can get it only by making the same stochastic action mistake or just an actual cycle.
		new_node_ptr = new_node_father
		old_node_ptr = old_node

		while new_node_ptr != None:
			new_visits[0] += new_node_ptr.num_visits
			new_visits[1] += 1
			new_node_ptr = self.tree.parent(new_node_ptr)

		while old_node_ptr != None: # getting to the old node wasn't necessarily through the current root. check all the way until None, the original root's parent.
			old_visits[0] += old_node_ptr.num_visits
			old_visits[1] += 1
			old_node_ptr = self.tree.parent(old_node_ptr)

		if new_visits[0] / new_visits[1] > old_visits[0] / old_visits[1]: # newer node is the more probable one. make the 1st path the invalid one: its the one that created the circle!
			old_node.invalid = True
			self.tree.update_id(old_id=old_node.identifier, new_id=hash((self.tree.parent(old_node).identifier, old_node.identifier)))
		else:
			new_node.invalid = True
			# it can be possible this is not the first time this scycle is detected, in which case we need to find a different way to hash because it was already hashed with the father id.
			new_node.identifier = hash((new_node_father.identifier, new_node.identifier))
			while new_node.identifier in self.tree.nodes.keys():
				new_node.identifier = hash((new_node_father.identifier, new_node.identifier))

	def is_parent_child_same(self, new_node, node):
		return new_node.pos[0] == node.pos[0] and new_node.pos[1] == node.pos[1] and new_node.state['direction'] == node.state['direction']

	def expand(self, node, depth):
		global newely_expanded
		action = node.untried_action()
		state, reward, terminated, truncated, _ = self.env.step(self.stochastic_action(action))
		done = terminated | truncated
		new_identifier = hash((tuple(node.pos), node.state['direction'], action, tuple(self.env.unwrapped.agent_pos), state['direction']))
		new_node = Node(identifier=new_identifier, state=state, action=action, action_space=self.action_space, reward=reward, terminal=done, pos=self.env.unwrapped.agent_pos, depth=depth)
		if new_node.pos[0] == 8 and new_node.pos[1] == 4:
			pass
		if new_identifier in self.tree.nodes.keys(): # who can tell which node is invalid? might be that this is the more probable way to get here, it just happened later. maybe think of summing back up the num of visits to decide which one to make invalid.
			if new_node.pos[0] == 8 and new_node.pos[1] == 4:
				pass
			# print("CIRCLE DETECTED!") # circle can be detected by 2 nodes making the wrong stochastic action one after another, in different times!
			self.decide_invalid_path(new_node_father=node, old_node=self.tree.nodes[new_identifier], new_node=new_node)
			# self.mark_invalid_children(node.children_identifiers, action)
		if self.is_parent_child_same(new_node, node): # this is not a regular circle but it indicates that the action - regardless if happened with or without intention, led to staying put. note this could happen even if the first if is true - twice in history someone tried to go against the wall from 2 different paths. both should be tagged invalid.
			if new_node.pos[0] == 8 and new_node.pos[1] == 4:
				pass
			new_node.invalid = True
			new_node.got_invalid = True
		self.tree.add_node(new_node, node)
		# if action == 2 and tuple(self.env.unwrapped.agent_pos) == tuple(node.pos): # if the new node is actually invalid, mark it along with the other nodes of the same action as invalid, meaning reward will be 0 for them.
		# 	self.mark_invalid_children(node.children_identifiers)
		newely_expanded += 1
		return new_node

	def stochastic_action(self, choice):
		prob_distribution = []
		actions = range(self.action_space)
		for action in actions:
			if action == choice: prob_distribution.append(PROB)
			else: prob_distribution.append(UNIFORM_PROB)
		return random.choices(actions, weights=prob_distribution, k=1)[0]

	def expand_selection_stochastic_node(self, node, resulting_identifier, terminated, truncated, reward, action, state, depth):
		global newely_expanded
		# the new node could result in a terminating state.
		done = terminated | truncated
		new_node = Node(resulting_identifier, state=state, action=action, action_space=self.action_space, reward=reward, terminal=done, pos=self.env.unwrapped.agent_pos, depth=depth)
		if resulting_identifier in self.tree.nodes.keys(): # We want to add the invalid node to propagate 0 reward to the fathers, so we must create a unique identifier: just add 1 currently.
			if new_node.pos[0] == 8 and new_node.pos[1] == 4:
				pass
			# print("CIRCLE DETECTED!")
			self.decide_invalid_path(new_node_father=node, old_node=self.tree.nodes[resulting_identifier], new_node=new_node)
		if self.is_parent_child_same(new_node, node): # this is not a regular circle but it indicates that the action - regardless if happened with or without intention, led to staying put.
			if new_node.pos[0] == 8 and new_node.pos[1] == 4:
				pass
			new_node.invalid = True
			new_node.got_invalid = True
		self.tree.add_node(new_node, node)
		newely_expanded += 1
		return new_node

	def simulation(self, node):
		if node.terminal:
			return node.reward
		if self.use_heuristic:
			# taken from Monte-Carlo Planning for Pathfinding in Real-Time Strategy Games , 2010.
			# need to handle the case of walking into a wall here: the resulting node will be considered invalid and it's reward and performance needs to be 0, but must handle stochasticity
			# suggestion to handle stochasticity - consider *all* the children associated with taking action 2 towards a wall as performance 0, even if they accidently led in walking to another direction.
			# which suggests the invalidity needs to be checked not according to the resulting state, rather according to the intended action itself and the environment! remember, you cannot access the "stochastic_action", it is meant to be hidden from you.
			if node.invalid: return 0
			else: return 1 / (abs(node.pos[0] - self.goal[0]) + abs(node.pos[1] - self.goal[1])) # large depth = less probability of obstacles -> larger nominator higher performance. further from goal -> larger denominator, lower performance.
		while True:
			action = random.randint(0, self.action_space-1)
			state, reward, terminated, truncated, _ = self.env.step(self.stochastic_action(action))
			done = terminated | truncated # this time there could be truncation unlike in the tree policy.
			if done:
				return reward

	def compute_value(self, parent, child, exploration_constant):
		exploration_term = exploration_constant * sqrt(2*log(parent.num_visits) / child.num_visits)
		return child.performance + exploration_term

	# return the best action from a node. the value of an action is the weighted sum of performance of all children that are associated with this action.
	def best_action(self, node, exploration_constant):
		tried_actions_values = {} # dictionary mapping actions to tuples of (cumulative number of visits of children, sum of (child performance * num of visits for child)) to compute the mean later
		if tuple(node.pos) == (1,2) and node.depth == 3 and node.action == 0:
			pass
		children = [child for child in self.tree.children(node) if not child.invalid]
		if not children: # all children are invalid. this node is invalid aswell.
			node.invalid = True
			node.got_invalid = True
			return -1
		for child in children:
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
			resulting_identifier = hash((tuple(node.pos), node.state['direction'], action, tuple(self.env.unwrapped.agent_pos), state['direction']))
			node = self.tree.nodes[resulting_identifier]
		return node, True

	# finds the ultimate path from the root node to a terminal state (the one that maximized rewards)
	def tree_policy(self, root_depth):
		node = self.tree.root
		depth = root_depth
		while not (node.terminal or node.invalid):
			depth += 1
			if self.tree.is_expandable(node):
				# expansion - in case there's an action that never been tried, its value is infinity to encourage exploration of all children of a node.
				return self.expand(node, depth), depth
			else:
				# selection - balance exploration and exploitation, coming down the tree - but note the selection might lead to new nodes because of stochaticity.
				best_action = self.best_action(node, exploration_constant=1/sqrt(2.0))
				if best_action == -1: break
				# important to simulate the env to get to some state, as the nodes don't hold this information.	
				state, reward, terminated, truncated, _ = self.env.step(self.stochastic_action(best_action))
				# due to stochasticity, nodes could sometimes be terminal and sometimes they aren't. important to update it. also, the resulting state
				# could be a state we've never been at due to uncertainty of actions' outcomes.
				# if the resulting state creates a parent-action-child triplet that hasn't been seen before, add to the tree and return it, similar result to 'expand'.
				# the hashing must include the action, because we want to enable getting to the same state stochastically from 2 different states: walking forward from (1,2) looking right and getting to (2,2) - the expected behavior, should be allowed even if the agent once stood at (2,1), looked down, turned right and accidently proceeded forward.
				resulting_identifier = hash((tuple(node.pos), node.state['direction'], best_action, tuple(self.env.unwrapped.agent_pos), state['direction']))
				if resulting_identifier not in node.children_identifiers and hash((node.identifier, resulting_identifier)) not in node.children_identifiers and hash((node.identifier,hash((node.identifier, resulting_identifier)))) not in node.children_identifiers: # The action was stochastic, enter only if it led to an unvisited state (checked by resulting identifier not existing and resulting identifier+1 too whic represents circle.
					return self.expand_selection_stochastic_node(node, resulting_identifier, terminated, truncated, reward, best_action, state, depth), depth
				# if got here, 1 of the 2 is correct: (1) the resulting state was seen once and we traverse it, or (2) we got to an invalid node, which caused a circle. In such case, the loop will be broken and a 0 reward will be returned.
				if hash((node.identifier, resulting_identifier)) in node.children_identifiers:
					node = self.tree.nodes[hash((node.identifier, resulting_identifier))] # loop will be broken and 0 reward will be returned later in simulation.
					assert node.invalid
				else:
					node = self.tree.nodes[resulting_identifier] # node is valid, next iteration should occur.
					# node could be invalid, but only if it got invalid throughout the iterations because he had no valid children, so its id was made regularaly.
					assert not node.invalid or node.got_invalid
    
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
			if best_action == -1:
				pass
			candidate_children = [child for child in self.tree.children(node) if child.action == best_action]
			node = max(candidate_children, key=lambda node: node.num_visits) # pick the child associated with the best action that was visited most, meaning it represents the desired action and not the stochasticity.
			trace.append(((node.state, node.pos), node.action))
		return trace
    

def save_model_and_generate_policy(tree, original_root, model_file_path, monteCarloTreeSearch):
	tree.root = original_root
	with open(model_file_path, 'wb') as file:  # Serialize the model
		monteCarloTreeSearch.env = None # pickle cannot serialize lambdas which exist in the env
		pickle.dump(monteCarloTreeSearch, file)
	return monteCarloTreeSearch.generate_policy_sequence()

def plan(env_name, problem_name, goal):
	global newely_expanded
	model_dir = get_model_dir(env_name=env_name, model_name=problem_name, class_name="MCTS")
	model_file_path = os.path.join(model_dir, "mcts_model.pth")
	if os.path.exists(model_file_path):
		print(f"Loading pre-existing mcts planner in {model_file_path}")
		with open(model_file_path, 'rb') as file:  # Load the pre-existing model
			monteCarloTreeSearch = pickle.load(file)
			return monteCarloTreeSearch.generate_policy_sequence()
	if not os.path.exists(model_dir): # if we reached here, the model doesn't exist. make sure its folder exists.
		os.makedirs(model_dir)
	steps = 8000
	print(f"No tree found. Executing MCTS, starting with {steps} rollouts for each action.")
	env = gym.make(id=problem_name)
	random.seed(2)
	tree = Tree()
	mcts = MonteCarloTreeSearch(env=env, tree=tree, goal=goal)
	original_root = tree.root
	plan = [] # a sequence of actions
	depth = 0
	while not tree.root.terminal: # we iterate until the root is a terminal state, meaning the game is over.
		max_reward = 0
		iteration = 0
		steps = max(1000,int(steps*0.9))
		print(f"Executing {steps} rollouts for each action now.")
		tq = tqdm(range(steps), postfix=f"Iteration: {iteration}, Num of steps: {len(plan)}. depth: {depth}. Max reward: {max_reward}. plan to {tuple(env.unwrapped.agent_pos)}, newely expanded: {0}")		
		for n in tq:
			iteration = n
			mcts.env.reset()
			# when executing the partial plan, it's possible the environment finished due to the stochasticity. the execution would return false if that happend.
			depth = len(plan)
			mcts.tree.root = original_root # need to return it to the original root before executing the partial plan as it can lead to a different path and the root can change between iterations.
			node, result = mcts.execute_partial_plan(plan)
			if not result:
				# false return value from partial plan execution means the plan is finished. we can mark our root as terminal and exit, happy with our plan.
				tree.root.terminal = True
				return save_model_and_generate_policy(tree=tree, original_root=original_root, model_file_path=model_file_path, monteCarloTreeSearch=mcts)
			plan_pos, plan_dir = node.pos, dict_dir_id_to_str[node.state['direction']]
			tree.root = node # determine the root to be the node executed after the plan for this iteration.
			node, depth = mcts.tree_policy(root_depth=depth) # find a path to a new unvisited node (unique sequence of actions) by utilizing explorative policy or choosing unvisited children recursively
			# if the node that returned from tree policy is terminal, the reward will be returned from "simulation" function immediately.
			reward = mcts.simulation(node) # proceed from that node randomly and collect the final reward expected from it (heuristic)
			if reward > max_reward:
				max_reward = reward
			mcts.backpropagation(node, reward) # update the performances of nodes along the way until the root
			tq.set_postfix_str(f"Iteration: {iteration}, Num of steps: {len(plan)}. depth: {depth}. Max reward: {max_reward}. plan to {tuple(plan_pos)}, looking {plan_dir}. newely expanded: {newely_expanded}")
		# update the root and start from it next time.
		newely_expanded = 0
		action = mcts.best_action(node=tree.root, exploration_constant=0)
		if action == -1:
			pass
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