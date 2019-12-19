from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT

from DQNAgent import DQNAgent

import numpy as np
import time
import os


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

# DISCOUNT = 0.99
# REPLAY_MEMORY_SIZE = 50_000 # How many last steps to keep for model training.
# MIN_REPLAY_MEMORY_SIZE = 1_000 # Minimum number of steps in a memory to start training.
# MINIBATCH_SIZE = 1 # How many steps (samples) to use for training.
# UPDATE_TARGET_EVERY = 5 # Terminal states (end of episodes).
# MODEL_NAME = "2x256"
# MIN_REWARD = -1000 # For model save.
# MEMORY_FRACTION = 0.20

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# Environment settings.
EPISODES = 1_000

# Exploration settings.
epsilon = 0.5 # Not a constant, will decay.
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Stats settings.
AGGREGATE_STATS_EVERY = 5 # Episodes.
SHOW_PREVIEW = True
ep_rewards = []

dim = (240, 256) # Dimensions of the state image.
left_crop = 0
top_crop = 0
right_crop = dim[1]
bottom_crop = dim[0]

# To load existing model, provide model="{model name}" as input to DQNAgent
agent = DQNAgent()
# Dfferentiate between training and evaluation
training = 1

go_to_cave = 0
state = env.reset()

if go_to_cave:
    go_to_start()

for ep in range(EPISODES):
    agent.tensorboard.step = ep
    min_ep_reward = 15
    max_ep_reward = -15
    ep_reward = 0
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
        if SHOW_PREVIEW and not ep % 1: #AGGREGATE_STATS_EVERY:
            env.render()
        if training:
            agent.update_replay_memory((observation, action, reward, new_observation, done))
            agent.train(done, step)
        observation = new_observation
        step += 1
        ep_reward += reward
        if reward < min_ep_reward:
            min_ep_reward = reward
        if reward > max_ep_reward:
            max_ep_reward = reward

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(ep_reward)
    if training and (not ep % AGGREGATE_STATS_EVERY or ep == 1):
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set threshold or a certain amount of episodes has passed
        if min_reward >= agent.MIN_REWARD or not ep % (1 * AGGREGATE_STATS_EVERY):
            average_reward = "%.2f" % average_reward
            min_reward = "%.2f" % min_reward
            max_reward = "%.2f" % max_reward
            # Serialize model to JSON
            model_json = agent.model.to_json()
            model_name = f"models/{agent.MODEL_NAME}__{int(time.time())}__{max_reward}max_{average_reward}avg_{min_reward}min"
            with open(f"{model_name}.json", 'w') as json_file:
                json_file.write(model_json)
            # Serialize weights to HDF5
            agent.model.save_weights(f"{model_name}.h5")
            print("Saved model:", model_name)

    print("Episode:", ep + 1, "\tep_reward:", "%.2f" % ep_reward, "\tmin_ep_reward:", "%.2f" % min_ep_reward, "\tmax_ep_reward:", "%.2f" % max_ep_reward, "\ttarget distance:", "%.2f" % info['target_distance'], "\tepsilon:", "%.2f" % epsilon)

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

env.close()
