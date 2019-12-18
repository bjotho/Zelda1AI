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
            self.children = children  # Dictionary {child.action_ix: child Node}, where each child Node of the current Node is listed
        # if children_qs is None:
        #     self.children_qs = {}
        # else:
        #     self.children_qs = children_qs # Dictionary {child.action_ix: Q-value} of Q-values for each child in self.children.
        self.id = id # Int
        self.action_ix = action_ix # Int corresponding to action (index) leading to this Node
        self.parent = parent # Node object referencing Node/state where action_ix leads to current Node
        # self.state = state # Multidimensional list
        # self.accumuated_reward = accumulated_reward # Int
        # self.done = done # Boolean

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
        self.HEAP_GC_FRAC = 0.2 # Fraction of used cells before garbage collecting the heap

    def get_length(self):
        return len(self.cells) - self.total_used

    # Check if hc1.score < hc2.score
    def is_lower(self, hc1, hc2):
        return hc1 < hc2

    def push(self, score, action_ix, parent_id):
        heapq.heappush(self.cells, HeapCell(False, score, action_ix, parent_id))

    # Pop the first unused HeapCell
    def pop_max(self):
        if self.get_length() == 0:
            print("Cannot pop empty heap!")
            return
        if (self.total_used / self.get_length()) > self.HEAP_GC_FRAC:
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
        if (self.total_used / self.get_length()) > self.HEAP_GC_FRAC:
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


# The tree contains the nodes (describing the structure) and any additional data per node
class Tree:
    def __init__(self, env, gamma, epsilon):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_id = 0
        self.root = None
        self.heap = Heap()
        self.objective = 0
        self.ep_done = False
        # These six lists are indexed by node.id (SOA style)
        self.nodes = [] # List of Node objects
        self.children_qs = [] # 2D List of Q-values for children nodes for each Node
        self.states = [] # List of game states/images. Contains RGB values for each pixel for each image
        self.sensors = [] # Sensor list. Each entry contains Link's (x, y) coordinates as well as the current map_location
        self.accumulated_rewards = [] # List of accumulated reward
        self.dones = [] # List of booleans

    def is_root(self, node):
        return node.parent is None

    def is_leaf(self, node):
        return len(node.children) == 0

    def all_children_explored(self, node):
        return len(node.children) == self.env.action_space.n

    def all_children_done(self, node):
        done = self.all_children_explored(node)
        for i in range(len(node.children)):
            if not self.dones[node.get_child_ix(i).id]:
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
        for node in self.nodes:
            leaf_list.append([])
            for qs in self.children_qs[node.id][:len(node.children)]:
                leaf_list[-1].append(self.accumulated_rewards[node.id] + self.gamma * qs)
        best_child_list = []
        for i in leaf_list:
            if len(i) > 0:
                best_child_list.append(np.max(i))
            else:
                best_child_list.append(None)
        best_child_parent_ix = 0
        tmp_max = best_child_list[0]
        for i, q in enumerate(best_child_list):
            if q is not None:
                if q > tmp_max:
                    tmp_max = q
                    best_child_parent_ix = i
        best_child = self.nodes[best_child_parent_ix].get_child_ix(np.argmax(leaf_list[best_child_parent_ix]))
        return best_child

    def get_tree_rank(self):
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
            node_reward = self.accumulated_rewards[node.id] - (self.accumulated_rewards[parent.id] if parent else 0)
            mn = np.mean(self.children_qs[nix]) if self.is_leaf(node) \
                else float(np.sum(avisitedc[nix].reshape(-1) * self.children_qs[nix])) / float(np.sum(avisitedc[nix]))
            if parent:
                self.children_qs[parent.id][node.action_ix] = node_reward + self.gamma * mn
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
            if parent:
                self.children_qs[parent.id][node.action_ix] = node_reward + self.gamma * mx
        # Update root
        self.dones[0] = self.all_children_done(self.root)

    def expand(self, parent_node, action_ix):
        state, reward, ep_done, info = self.env.step(action_ix)
        sensors = (info['x_pos'], info['y_pos'], info['map_location'])
        self.env.render()
        if ep_done:
            self.objective = 0
            self.ep_done = True
        done = self.check_objective_completed(info['objective'], ep_done)
        if done:
            children_qs = np.zeros(self.env.action_space.n)
            return
        else:
            children_qs = agent.get_qs(state)

        accumulated_reward = self.accumulated_rewards[parent_node.id] + reward
        self.add_node(parent_node, action_ix)
        self.sensors.append(sensors)
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

    def build_tree(self, max_size, root_state, root_sensors):
        # Build the root node
        self.add_node()
        self.root = self.nodes[0]
        root_children_qs = agent.get_qs(root_state)

        # Update the tree with root
        self.sensors.append(root_sensors)
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
                self.env.reset()
                self.ep_done = False

            # Root is done or heap is empty
            if self.dones[0] or self.heap.get_length() == 0:
                return

            # Choose parent and action
            action_ix, parent_id = self.heap.pop_max() if np.random.random() > self.epsilon else self.heap.pop_rand()
            assert not self.dones[parent_id]
            parent = self.nodes[parent_id]
            # assert action_ix in parent.children.keys()

            # Add a new node to the tree
            self.expand(parent, action_ix)

    def add_node(self, parent=None, action_ix=None):
        if action_ix is None:
            if parent is None:
                action_ix = self.env.action_space.sample()
            else:
                sibling_actions = list(parent.children.keys())
                action_ix = self.env.action_space.sample()
                if len(sibling_actions) >= self.env.action_space.n:
                    action_ix = len(sibling_actions)
                while action_ix in sibling_actions:
                    action_ix = self.env.action_space.sample()

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


#w,ll = trainit(N, states, qs, LR, batchsize, priorities)

# def trainit(N, states, mqs, lr, batchsize, priorities):
#     h = np.sort(priorities)[int(np.floor(0.9 * len(priorities)))]
#     _p = [np.max([h, p]) for p in priorities]
#     # _at = np.random.choice(priorities, num_of_elements_in_output_list, _p)
#     tll = 0.0
#     for j in range(N):
#         x, y, bix = batchit(states, mqs, batchsize, _p)
#
# def batchit(states, qs, batchsize, probabilities):
#     assert len(qs) > 0
#     assert len(states) == len(qs)
#     NACT = len(qs[0])
#     x = np.zeros(shape=(len(states[0]), len(states[0][0]), 1, batchsize))
#     y = np.zeros(shape=(NACT, batchsize))
#     bix = np.random.rand(np.random.choice([i for i in range(batchsize)], 1, p=probabilities), batchsize)


env = gym_zelda_1.make('Zelda1-v0')
env = JoypadSpace(env, MOVEMENT)
state = env.reset()
root_sensors = (119, 120, 141)

EPISODES = 1_000

# Tree parameters
max_tree_size = 500
epsilon = 0.5
MIN_EPSILON = 0.01
fac = 0.995
gamma = 0.98

# Training parameters
LR = 0.01
batchsize = 64
train_epochs = 20
max_states = 80_000
train_when_min = 1_000

# Validation parameter
val_safey = 1.0

# Experience buffer
states = []
qs = []
priorities = []

# For stats
rewards = []
ranks = []
num_of_done_leaves = []
num_of_leaves = []
val_steps = []
val_rewards = []
avg_window = 50
weighted_nodes_threshold = 200

agent = DQNAgent()

for ep in range(EPISODES):
    #### Accumulating tree ###################################################################
    print("------------------------------------------------------------------\nEpisode", ep+1)

    tree = Tree(env, gamma, epsilon)
    tree.build_tree(max_tree_size, state, root_sensors)
    tree.backprop_weighted_q(weighted_nodes_threshold)

    max_rank = tree.get_tree_rank()
    ranks.append(max_rank)
    rewards.append(tree.get_accumulated_reward())
    num_of_done_leaves.append(np.sum([tree.dones[n.id] for n in tree.nodes if tree.is_leaf(n)]))
    num_of_leaves.append(np.sum([tree.is_leaf(n) for n in tree.nodes]))
    ixs = [n.id for n in tree.nodes if (not tree.is_leaf(n) or tree.dones[n.id])]
    for ix in ixs:
        qs.append(tree.children_qs[ix])
        states.append(tree.states[ix])

    if len(priorities) == 0:
        f = 1.0
    else:
        f = np.median(priorities)

    for i in (f * np.ones(len(ixs))):
        priorities.append(i)
    if len(qs) > max_states:
        qs = qs[len(qs)-max_states:]
        states = states[len(states)-max_states:]
        priorities = priorities[len(priorities)-max_states:]

    #### Training ###########################################################################
    if len(qs) >= train_when_min:
        N = round(len(ixs) * train_epochs / batchsize)
        print("---- training for", N, "iterations ----")
        agent.trainit(N, states, qs, LR, batchsize, priorities)
        if epsilon > MIN_EPSILON:
            epsilon *= fac

    print("Tree stats:\n\tmax_rank:", ranks[-1], "\trewards:", "%.2f" % rewards[-1], "\tdone_leaves:", num_of_done_leaves[-1], "\tleaves:", num_of_leaves[-1])
    print("General stats:\n\tlen(qs):", len(qs), "\tlen(ixs):", len(ixs), "\tmin(priorities):", np.min(priorities))
    print(tree)















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
