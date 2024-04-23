import random
from math import sqrt, log
from utils.node import Node
from utils.tree import Tree
from gym.envs.registration import register
import gym

class MonteCarloTreeSearch():

    def __init__(self, env, tree):
        self.env = env
        self.tree = tree
        self.action_space = self.env.action_space.n
        state = self.env.reset()
        self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, reward=0, terminal=False))

    def expand(self, node):
        action = node.untried_action()
        state, reward, done, _ = self.env.step(action)
        new_node = Node(state=state, action=action, action_space=self.action_space, reward=reward, terminal=done)
        self.tree.add_node(new_node, node)
        return new_node

    def default_policy(self, node):
        if node.terminal:
            return node.reward

        while True:
            action = random.randint(0, self.action_space-1)
            state, reward, done, _ = self.env.step(action)
            if done:
                return reward

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

    def tree_policy(self):
        node = self.tree.root
        while not node.terminal:
            if self.tree.is_expandable(node):
                return self.expand(node)
            else:
                node = self.best_child(node, exploration_constant=1.0/sqrt(2.0))
                state, reward, done, _ = self.env.step(node.action)
                assert node.state == state
        return node

    def backward(self, node, value):
        while node:
            node.num_visits += 1
            node.total_simulation_reward += value
            node.performance = node.total_simulation_reward/node.num_visits
            node = self.tree.parent(node)

    def forward(self):
        self._forward(self.tree.root)

    def _forward(self,node):
        best_child = self.best_child(node, exploration_constant=0)

        print("****** {} ******".format(best_child.state))

        for child in self.tree.children(best_child):
            print("{}: {:0.4f}".format(child.state, child.performance))

        if len(self.tree.children(best_child)) > 0:
            self._forward(best_child)
            
def init_env():
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False}
    )
    return gym.make('FrozenLakeNotSlippery-v0')
            
def plan():
    random.seed(2)
    env = init_env()
    tree = Tree()
    monteCarloTreeSearch = MonteCarloTreeSearch(env=env, tree=tree)
    steps = 10000

    for _ in range(0, steps):
        env.reset()
        node = monteCarloTreeSearch.tree_policy()
        reward = monteCarloTreeSearch.default_policy(node)
        monteCarloTreeSearch.backward(node, reward)

    monteCarloTreeSearch.tree.show()
    monteCarloTreeSearch.forward()