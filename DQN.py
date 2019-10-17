from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT

from DQNAgent import DQNAgent

import numpy as np


def read_NES_palette():
    palette = []
    with open("NES_Palette.txt", 'r') as f:
        line = f.readline()
        while line is not "":
            palette.append([])
            rgb_string = [x.strip() for x in line.split(',')]
            for i in rgb_string:
                palette[-1].append(int(i))
            line = f.readline()
    return palette


def go_to_start():
    if start_in_level_1:
        # Go get sword for agent
        a = []
        for i in range(30):
            a.append(5)     # up
        for i in range(45):
            a.append(4)     # left
        for i in range(30):
            a.append(5)     # up
        # for i in range(120):
        #     a.append(5)     # up
        # for i in range(70):
        #     a.append(4)     # left
        # for i in range(38):
        #     a.append(5)     # up
        # for i in range(30):
        #     a.append(4)     # left

        global highest_objective
        # global Q
        # Start x and y pos: (120, 141)
        for step in range(len(a)):
            if step < len(a) - 1:
                state, reward, done, info = env.step(a[step])
                # if highest_objective < info['objective']:
                    # Q = np.append(Q, np.random.uniform(low=-15, high=15, size=([1, 16, 11, len(MOVEMENT)])), axis=0)
                    # highest_objective = info['objective']
            else:
                state, reward, done, info = env.step(0)
            # env.render()
        # print("Agent taking over")


# def get_discrete_state(state):
#     x_i = state[0] // 16
#     y_i = (state[1] - 61) // 16
#     return tuple((x_i, y_i))


# def discrete_observation(state, left_crop=0, top_crop=0, right_crop=None, bottom_crop=None):
#     """Basically crops the input image (state) to the specified pixels"""
#     observation = []
#     for i in state[top_crop:(bottom_crop if bottom_crop < dim[0] else None)]:
#         observation.append([])
#         for j in i[left_crop:(right_crop if right_crop < dim[1] else None)]:
#             observation[-1].append([])
#             for k in j:
#                 observation[-1][-1].append(k)
#
#     return np.array(observation)


env = gym_zelda_1.make('Zelda1-v0')
env = JoypadSpace(env, MOVEMENT)
# The area where Link can be is approximately 255*175 pixels (x:0-255, y:64-239).
# If we divide these dimensions by 16, we get a (16, 11) matrix. This matrix will represent each discrete position Link can be in,
# and for each of these discrete positions, he can perform len(MOVEMENT) distinct actions. Therefore, the Q matrix will have the dimensions [11,16,len(MOVEMENT)].
# Q = np.random.uniform(low=-15, high=15, size=([1, 16, 11, len(MOVEMENT)]))
# print(Q.shape)

start_in_level_1 = 0
state = env.reset()

# DISCOUNT = 0.99
# REPLAY_MEMORY_SIZE = 50_000 # How many last steps to keep for model training.
# MIN_REPLAY_MEMORY_SIZE = 1_000 # Minimum number of steps in a memory to start training.
# MINIBATCH_SIZE = 1 # How many steps (samples) to use for training.
# UPDATE_TARGET_EVERY = 5 # Terminal states (end of episodes).
# MODEL_NAME = "2x256"
# MIN_REWARD = -1000 # For model save.
# MEMORY_FRACTION = 0.20

# Environment settings.
EPISODES = 100

# Exploration settings.
epsilon = 0.5 # Not a constant, will decay.
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Stats settings.
AGGREGATE_STATS_EVERY = 1 # Episodes.
SHOW_PREVIEW = True

dim = (240, 256) # Dimensions of the state image.
left_crop = 0
top_crop = 0
right_crop = dim[1]
bottom_crop = dim[0]
agent = DQNAgent()

if go_to_start:
    go_to_start()

for ep in range(EPISODES):
    agent.tensorboard.step = ep
    min_reward = 15
    max_reward = -15
    step = 1
    state = env.reset()
    info = None
    observation = state #discrete_observation(state, x_cutoff_start, y_cutoff_start, x_cutoff_end, y_cutoff_end) # Remove "status bar" from observation
    # print("observation.shape:", observation.shape)
    done = False
    while not done:
        if np.random.random() < epsilon: # Decide whether to perform random action according to epsilon
            action = env.action_space.sample()
        else:
            action = np.argmin(agent.get_qs(observation))
        state, reward, done, info = env.step(action)
        new_observation = state #discrete_observation(state, x_cutoff_start, y_cutoff_start, x_cutoff_end, y_cutoff_end)
        # x_cutoff_start = info['x_pos'] - 32
        # y_cutoff_start = info['y_pos'] - 32
        # x_cutoff_end = info['x_pos'] + 32
        # y_cutoff_end = info['y_pos'] + 32
        if SHOW_PREVIEW and not ep % AGGREGATE_STATS_EVERY:
            env.render()
        agent.update_replay_memory((observation, action, reward, new_observation, done))
        agent.train(done, step)
        observation = new_observation
        step += 1
        if reward < min_reward:
            min_reward = reward
        if reward > max_reward:
            max_reward = reward
    print("Episode:", ep + 1, "\tmin_reward:", "%.2f" % min_reward, "\tmax_reward:", "%.2f" % max_reward,
          "\ttarget distance:", "%.2f" % info['target_distance'], "\tepsilon:", "%.2f" % epsilon)

env.close()
