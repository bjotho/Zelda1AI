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
            self.children = {}
        else:
            self.children = children  # Dictionary {Int, Node}, where each child Node of the current Node is listed
        self.id = id # Int
        self.action_ix = action_ix # Int corresponding to action (index) leading to this Node
        #self.R = action_r
        #self.C = accumulated_past_reward
        #self.Q = q
        self.parent = parent # Node object referencing Node/state where action_ix leads to current Node
        # self.done = done

    def add_child(self, node, action_ix):
        self.children[action_ix] = node

    def get_child_aix(self, action_ix):
        return self.children[action_ix]

    def get_child_ix(self, ix):
        return self.children[list(self.children.keys())[ix]]

    def show_tree(self, recursion_level=0):
        output = ""
        for i in range(recursion_level):
            output += "\t"
        output += "Node " + str(self.id) + "\n"
        for i in range(len(self.children)):
            output += self.get_child_ix(i).show_tree(recursion_level+1)
        return output

    def __repr__(self):
        output = "Node: " + str(self.id) + "\n{\n\taction_ix: " + str(self.action_ix)
        output += "\n\tparent: Node " + str(self.parent.id) if self.parent else "\n\tparent: None"
        output += "\n\tno. children: " + str(len(self.children))
        if len(self.children) > 0:
            output += "\n\t{"
            for i in range(len(self.children)):
                output += "\n\t\t" + str(self.get_child_ix(i).action_ix) + ":\tNode " + str(self.get_child_ix(i).id)
            output += "\n\t}"
        output += "\n}"
        return output


class HeapCell:
    def __init__(self, is_used, score, action_ix, parent_id):
        self.is_used = is_used # because we cannot efficiently pop a random element from a heap
        self.score = score
        self.action_ix = action_ix
        self.parent_id = parent_id

    def __lt__(self, other):
        return self.score < other.score

    def __le__(self, other):
        return self.score <= other.score


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
    def __init__(self, gamma, epsilon):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_id = 0
        self.root = None
        self.heap = Heap()
        self.objective = 0
        self.ep_done = False
        # These five lists are indexed by node.id (SOA style)
        self.nodes = [] # List of Node objects
        self.children_qs = [] # 2D List of Q-values for children nodes for each Node
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
        for i in range(len(node.children)):
            if not self.dones[node.get_child_ix[i].id]:
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

    def get_best_leaf(self):
        leaf_list = []
        for i, qs in enumerate(self.children_qs):
            leaf_list.append(self.accumulated_rewards[i] + self.gamma * qs)
        return self.nodes[int(np.argmax(leaf_list))]

    def get_rank_tree(self):
        return self.get_rank(self.get_best_leaf())

    def get_accumulated_reward(self):
        return np.max(self.accumulated_rewards)

    # Calculate number of times each node was visited
    # Iterate in reverse, so parents have all children already updated when we get to them
    def calc_visitedc(self, maxval=-1):
        visitedc = np.zeros(len(self.nodes))
        for nix in range(len(self.nodes)-1, -1, -1):
            node = self.nodes[nix]
            for key, ch in node.children.items():
                visitedc[nix] += visitedc[ch.id]
                if 0 < maxval < visitedc[nix]:
                    visitedc[nix] = maxval
            visitedc[nix] += 1
        return visitedc

    def calc_action_visitedc(self, visited_threshold=-1, nonexplored_value=0):
        visitedc = self.calc_visitedc(visited_threshold)
        avc = []
        for node in self.nodes:
            _avc = nonexplored_value * np.ones(len(self.children_qs[0]))
            for ix, ch in node.children.items():
                _avc[ix] = visitedc[ch.id]
            avc.append(_avc)
        return avc

    # Calculate weighted Qs, root is done separately
    # Iterate in reverse, so parents have all children already updated when we get to them
    def backprop_weighted_q(self, visited_threshold=-1):
        avisitedc = np.array(self.calc_action_visitedc(visited_threshold))
        for nix in range(len(self.nodes)-1, -1, -1):
            node = self.nodes[nix]
            self.dones[node.id] = self.all_children_done(node)
            # Update Q at parent
            parent = node.parent
            node_reward = self.accumulated_rewards[node.id] - self.accumulated_rewards[parent.id]
            mn = np.mean(self.children_qs[nix]) if self.is_leaf(node) \
                else float(np.sum(avisitedc[nix].reshape(-1) * self.children_qs[nix])) / float(np.sum(avisitedc[nix]))
            self.children_qs[parent][node.action_ix] = node_reward + self.gamma * mn
        # Update root
        self.dones[0] = self.all_children_done(self.root)

    # Backprop everybody in reverse, so parents have all children already updated when we get to them
    def backprop_max_q(self):
        for nix in range(len(self.nodes)-1, -1, -1):
            node = self.nodes[nix]
            parent = node.parent
            self.dones[node.id] = self.all_children_done(node)
            # Update Q at parent
            mx = np.max(self.children_qs[node.id])
            node_reward = self.accumulated_rewards[node.id] - self.accumulated_rewards[parent.id]
            self.children_qs[parent.id][node.action_ix] = node_reward + self.gamma * mx
        # Update root
        self.dones[0] = self.all_children_done(self.root)

    def expand(self, parent_node, action_ix):
        state, reward, ep_done, info = env.step(action_ix)
        env.render()
        if ep_done:
            self.objective = 0
            self.ep_done = True
        done = self.check_objective_completed(info['objective'], ep_done)
        if done:
            children_qs = np.zeros(env.action_space.n)
        else:
            children_qs = agent.get_qs(state)
        self.add_node(parent_node, action_ix)
        accumulated_reward = self.accumulated_rewards[parent_node.id] + reward
        self.children_qs.append(children_qs)
        self.states.append(state)
        self.accumulated_rewards.append(accumulated_reward)
        self.dones.append(done)
        if not done:
            for aix, q in enumerate(self.children_qs[parent_node.id]):
                score = accumulated_reward + self.gamma * q
                self.heap.push(score, aix, self.nodes[-1].id)

    def check_objective_completed(self, objective, done):
        if self.objective < objective:
            self.objective += 1
            return True

        return False

    def build_tree(self, max_size, root_state):
        # Build the root node
        self.add_node()
        self.root = self.nodes[0]
        root_children_qs = agent.get_qs(root_state)

        # Update the tree with root
        self.children_qs.append(root_children_qs)
        self.states.append(root_state)
        self.accumulated_rewards.append(0.0)
        self.dones.append(False)
        for aix, q in enumerate(self.children_qs[0]):
            score  = self.gamma * q
            self.heap.push(score, aix, self.root.id)

        # Add new nodes in a loop
        for i in range(max_size):
            if self.ep_done:
                env.reset()
                self.ep_done = False

            # Root is done or heap is empty
            if self.dones[0] or self.heap.get_length() == 0:
                return

            # Choose parent and action
            action_ix, parent_id = self.heap.pop_max() if np.random.random() > self.epsilon else self.heap.pop_rand()
            parent = self.nodes[parent_id]

            # Add a new node to the tree
            self.expand(parent, action_ix)

    def add_node(self, parent=None, action_ix=None):
        if action_ix is None:
            if parent is None:
                action_ix = env.action_space.sample()
            else:
                sibling_actions = list(parent.children.keys())
                action_ix = env.action_space.sample()
                if len(sibling_actions) >= env.action_space.n:
                    action_ix = len(sibling_actions)
                while action_ix in sibling_actions:
                    action_ix = env.action_space.sample()

        new_node = Node(self.running_id, action_ix, parent)
        if parent is not None:
            parent.add_child(new_node, action_ix)
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
                parent_gen_2 = self.root.get_child_ix(i)
                self.add_node(parent_gen_2)
                for k in range(np.random.randint(0, 3)):
                    parent_gen_3 = parent_gen_2.get_child_ix(j)
                    self.add_node(parent_gen_3)

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
epsilon = 0.25
agent = DQNAgent()
tree = Tree(gamma, epsilon)
# tree.build_example_tree(1)
tree.build_tree(5500, state)


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
