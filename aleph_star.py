from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT
from DQNAgent import DQNAgent

import numpy as np
import heapq


# a Node contains the minimum needed to hold the tree structure
# a Node is effectively the representation of a given state
class Node:
    def __init__(self, id, action_ix, parent, children=None):
        if children is None:
            children = {}
        self.id = id # Int
        self.action_ix = action_ix # Int corresponding to action (index) leading to this Node
        #self.R = action_r
        #self.C = accumulated_past_reward
        #self.Q = q
        self.parent = parent # Node object referencing Node/state where action_ix leads to current Node
        self.children = children # Dictionary {Int, Node}, where each child Node of the current Node is listed
        #self.done = done

    def add_child(self, node):
        self.children[len(self.children)] = node

    def show_tree(self, recursion_level=0):
        output = ""
        for i in range(recursion_level):
            output += "\t"
        output += "Node " + str(self.id) + "\n"
        for i in range(len(self.children)):
            output += self.children[i].show_tree(recursion_level+1)
        return output

    def __repr__(self):
        output = "Node: " + str(self.id) + "\n{\n\taction_ix: " + str(self.action_ix)
        if self.parent is None:
            output += "\n\tparent: None"
        else:
            output += "\n\tparent: Node " + str(self.parent.id)
        output += "\n\tno. children: " + str(len(self.children))
        if len(self.children) > 0:
            output += "\n\t{"
            for i in range(len(self.children)):
                output += "\n\t\t" + str(i) + ":\tNode " + str(self.children[i].id)
            output += "\n\t}"
        output += "\n}"
        return output


class HeapCell:
    def __init__(self, is_used, score, action_ix, parent_id):
        self.is_used = is_used # because we cannot efficiently pop a random element from a heap
        self.score = score
        self.action_ix = action_ix
        self.parent_id = parent_id


class Heap:
    def __init__(self):
        self.cells = [] # List of HeapCell()
        self.total_used = 0

    def get_length(self):
        return len(self.cells) - self.total_used

    # Return the HeapCell with lowest score
    # NB!!! Julia implementation returns boolean testing if hc1 < hc2
    def get_lowest(self, hc1, hc2):
        hc_min = np.argmin([hc1.score, hc2.score])
        if hc_min == 0:
            return hc1
        else:
            return hc2

    def push(self, score, action_ix, parent_id):
        heapq.heappush(self.cells, HeapCell(False, score, action_ix, parent_id))

    # Pop the first unused HeapCell
    def pop_max(self):
        if self.get_length() == 0:
            print("Cannot pop empty heap!")
            return
        if (self.total_used / self.get_length()) > HEAP_GC_FRAC:
            self.garbage_collect()
        while True:
            hc = heapq.heappop(self.cells)
            if hc.is_used:
                self.total_used -= 1
                continue
            return hc.action_ix, hc.parent_id

    # Pop random unused HeapCell
    def pop_rand(self):
        if self.get_length() == 0:
            print("Cannot randomly pop empty heap!")
        if (self.total_used / self.get_length()) > HEAP_GC_FRAC:
            self.garbage_collect()
        while True:
            ix = np.random.randint(0, self.get_length())
            hc = self.cells[ix]
            if hc.is_used:
                continue
            else:
                # Mark hc as used
                hc = HeapCell(True, hc.score, hc.action_ix, hc.parent_id)
                self.total_used += 1
                return hc.action_ix, hc.parent_id

    # Remove HeapCells marked with is_used from the Heap
    def garbage_collect(self):
        tmp = []
        for hc in self.cells:
            if hc.is_used:
                continue
            else:
                heapq.heappush(tmp, hc)
        self.cells = tmp
        self.total_used = 0


# The tree contains the nodes (describing the structure)
# and any additional data per node
class Tree:
    def __init__(self, gamma):
        self.gamma = gamma
        self.running_id = 0
        self.root = None
        self.heap = Heap()
        # These six lists are indexed by node.id (SOA style)
        self.nodes = [] # List of Node objects
        self.sensors = [] # List of sensor readings
        self.children_qs = [] # List of Q-values for children nodes
        self.states = [] # State list
        self.accumulated_rewards = [] # List of accumulated reward
        self.dones = [] # List of booleans

    def is_root(self, node):
        return node.parent is None

    def is_leaf(self, node):
        return len(node.children) == 0

    def all_children_explored(self, node):
        return len(node.children) == env.action_space.n

    def all_children_done(self, node):
        done = self.all_children_explored(node)
        for key, child in node.children:
            if not self.dones[child.id]:
                done = False
        return done

    def get_rank(self, node):
        rank = 1
        while True:
            if node.parent is None:
                break
            node = node.parent
            rank += 1
        return rank

    def build_tree(self, agent, state):
        # Build the root node
        root_state = state
        root_children_qs = agent.get_qs(state)
        ACTIONC = len(root_children_qs)
        root_action_ix = env.action_space.sample()
        root_id = 1
        root_accumulated_reward = 0.0
        root_done = False
        self.add_node()
        self.root = self.nodes[0]

        # Build the tree
        #nodes = a list of Node() objects
        #states = a list of states
        #children_qs = a list of lists of Q-values for children nodes
        #accumulated_rewards = list of total accumulated reward at each node
        #dones = list of booleans

        # print(self.root)
        # print(self.is_root(self.root))
        # print(self.is_root(self.root.children[0]))

    def add_node(self, parent=None, action_ix=None):
        if action_ix is None:
            action_ix = env.action_space.sample()
        new_node = Node(self.running_id, action_ix, parent)
        if parent is not None:
            parent.add_child(new_node)
        self.nodes.append(new_node)
        self.running_id += 1

    def build_example_tree(self, print_nodes):
        if len(self.nodes) > 0:
            print("Cannot build example tree, tree is not empty!")
            return

        print("Example tree")

        self.add_node()
        self.root = self.nodes[0]

        for i in range(np.random.randint(2, 4)):
            self.add_node(self.root)
            for j in range(np.random.randint(1, 5)):
                self.add_node(self.root.children[i])
                for k in range(np.random.randint(0, 3)):
                    self.add_node(self.root.children[i].children[j])

        if print_nodes:
            print("")
            for node in self.nodes:
                print(node)

        print("")
        print(self)

    def __repr__(self):
        return self.root.show_tree()



env = gym_zelda_1.make('Zelda1-v0')
env = JoypadSpace(env, MOVEMENT)
state = env.reset()

HEAP_GC_FRAC = 0.2 # Fraction of used cells before garbage collecting the heap
gamma = 0.9

agent = DQNAgent()
tree = Tree(gamma)
tree.build_example_tree(0)
for node in tree.nodes:
    print("Node", node.id, ":", tree.all_children_explored(node))



# # Environment settings.
# EPISODES = 100
#
# # Exploration settings.
# epsilon = 1 # Not a constant, will decay.
# EPSILON_DECAY = 0.99975
# MIN_EPSILON = 0.001
#
# # Stats settings.
# SHOW_PREVIEW = True
#
# dim = (240, 256) # Dimensions of the state image.
# left_crop = 0
# top_crop = 0
# right_crop = dim[1]
# bottom_crop = dim[0]
#
# for ep in range(EPISODES):
#     min_reward = 15
#     max_reward = -15
#     step = 1
#     state = env.reset()
#     info = None
#     done = False
#     while not done:
#         if np.random.random() < epsilon: # Decide whether to perform random action according to epsilon
#             action = env.action_space.sample()
#         else:
#             # implement aleph star
#             action = env.action_space.sample()
#         state, reward, done, info = env.step(action)
#         new_state = state
#         if SHOW_PREVIEW and not ep % 1:
#             env.render()
#         state = new_state
#         step += 1
#         if reward < min_reward:
#             min_reward = reward
#         if reward > max_reward:
#             max_reward = reward
#     print("Episode:", ep + 1, "\tmin_reward:", "%.2f" % min_reward, "\tmax_reward:", "%.2f" % max_reward,
#           "\ttarget distance:", "%.2f" % info['target_distance'], "\tepsilon:", "%.2f" % epsilon)
#
# env.close()
