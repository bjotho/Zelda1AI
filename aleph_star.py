from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT

import numpy as np


# def discrete_observation(state, left_crop=0, top_crop=0, right_crop=None, bottom_crop=None):
#     """Basically crops the input image (state) to the specified pixels"""
#     observation = []
#     observation = state[top_crop:(bottom_crop if bottom_crop < dim[0] else None)]
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

map_graph = {}

# DISCOUNT = 0.99
# MIN_REWARD = -1000 # For model save.
# MEMORY_FRACTION = 0.20
#
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
#             # TODO: implement aleph star
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
