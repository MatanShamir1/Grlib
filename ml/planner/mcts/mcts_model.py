import random
from math import sqrt, log
from utils.node import Node
from utils.tree import Tree
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
        self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, reward=0, terminal=False, pos=self.env.unwrapped.agent_pos))

    def expand(self, node):
        action = node.untried_action()
        state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated | truncated
        new_node = Node(state=state, action=action, action_space=self.action_space, reward=reward, terminal=done, pos=self.env.unwrapped.agent_pos)
        self.tree.add_node(new_node, node)
        return new_node

    def random_policy(self, node):
        if node.terminal:
            return node.reward

        while True:
            action = random.randint(0, self.action_space-1)
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            if done:
                return reward
                if reward != 0:
                    x=2

    def compute_value(self, parent, child, exploration_constant):
        exploitation_term = child.total_simulation_reward / child.num_visits
        exploration_term = exploration_constant * sqrt(2 * log(parent.num_visits) / child.num_visits)
        return exploitation_term + exploration_term

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

    # finds the ultimate path from the root node to a terminal state (the one that maximized rewards)
    def tree_policy(self):
        node = self.tree.root
        while not node.terminal:
            if self.tree.is_expandable(node):
                return self.expand(node)
            else:
                node = self.best_child(node, exploration_constant=1.0/sqrt(2.0))
                state, reward, terminated, truncated, _ = self.env.step(node.action)
                done = terminated | truncated
                # assert np.all(node.state == state)
        return node

    # receives a final state node and updates the rewards of all the nodes on the path to the root
    def backward(self, node, value):
        while node:
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
    
    def generate_policy_sequence(self, is_cont=False):
        trace = []
        node = self.tree.root
        while len(self.tree.children(node)) > 0:
            next = self.best_child(node, exploration_constant=0)
            if is_cont: trace.append((node.state, next.action))
            else: trace.append((node.pos, next.action))
            node = next
        if is_cont: trace.append((node.state, None))
        else: trace.append((node.pos, None))
        return trace
            
def plan(problem_name):
    random.seed(2)
    env = gym.make(id=problem_name)
    tree = Tree()
    monteCarloTreeSearch = MonteCarloTreeSearch(env=env, tree=tree)
    steps = 8000

    
    for _ in range(0, steps):
        env.reset()
        node = monteCarloTreeSearch.tree_policy() # find a path to a new unvisited node state by utilizing explorative policy or choosing unvisited children recursively
        reward = monteCarloTreeSearch.random_policy(node) # proceed from that node randomly and collect the final reward expected from it (heuristic)
        monteCarloTreeSearch.backward(node, reward)  # update the performances of nodes along the way

    print(monteCarloTreeSearch.generate_policy_sequence(is_cont=False))
    
if __name__ == "__main__":
    register(
        id="MiniGrid-DynamicGoalEmpty-8x8-3x6-v0",
        entry_point="minigrid.envs:DynamicGoalEmpty",
        kwargs={"size": 8, "agent_start_pos" : (1, 1), "goal_pos": (3,6) },
    )
    plan("MiniGrid-DynamicGoalEmpty-8x8-3x6-v0")