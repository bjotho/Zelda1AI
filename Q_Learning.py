from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT
import numpy as np


def go_to_start():
    if start_in_level_1:
        # Go get sword for agent
        a = []
        for i in range(30):
            a.append(5)  # up
        for i in range(45):
            a.append(4)  # left
        for i in range(200):
            a.append(5)  # up
        # for i in range(120):
        #     a.append(5)     # up
        # for i in range(70):
        #     a.append(4)     # left
        # for i in range(38):
        #     a.append(5)     # up
        # for i in range(30):
        #     a.append(4)     # left

        global highest_objective
        global Q
        # Start x and y pos: (120, 141)
        for step in range(len(a)):
            if step < len(a) - 1:
                state, reward, done, info = env.step(a[step])
                if highest_objective < info['objective']:
                    Q = np.append(Q, np.random.uniform(low=-15, high=15, size=([1, 16, 11, len(MOVEMENT)])), axis=0)
                    highest_objective = info['objective']
            else:
                state, reward, done, info = env.step(0)
            # env.render()
        # print("Agent taking over")


def get_discrete_state(state):
    x_i = state[0] // 16
    y_i = (state[1] - 61) // 16
    return tuple((x_i, y_i))


env = gym_zelda_1.make('Zelda1-v0')
env = JoypadSpace(env, MOVEMENT)
# The area where Link can be is approximately 255*175 pixels (x:0-255, y:64-239).
# If we divide these dimensions by 16, we get a (16, 11) matrix. This matrix will represent each discrete position Link can be in,
# and for each of these discrete positions, he can perform len(MOVEMENT) distinct actions. Therefore, the Q matrix will have the dimensions [16,11,len(MOVEMENT)].
Q = np.random.uniform(low=-15, high=15, size=([1, 16, 11, len(MOVEMENT)]))

start_in_level_1 = 0
state = env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20_000

# Exploration settings.
epsilon = 0.5 # Not a constant, will decay.
EPSILON_DECAY = 0.975
MIN_EPSILON = 0.01

highest_objective = 0

for ep in range(EPISODES):
    ep_reward = 0
    min_reward = 15
    max_reward = -15
    t = 0
    state = env.reset()
    if start_in_level_1:
        go_to_start()
    done = False
    state, reward, done, info = env.step(0)
    discrete_state = get_discrete_state((info['x_pos'], info['y_pos']))
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmin(Q[info['objective']][get_discrete_state((info['x_pos'], info['y_pos']))])
        new_state, reward, done, info = env.step(action)
        new_discrete_state = get_discrete_state((info['x_pos'], info['y_pos']))
        if ep % 1 == 0:
            env.render()
        if not done:
            if highest_objective < info['objective']:
                Q = np.append(Q, np.random.uniform(low=-15, high=15, size=([1, 16, 11, len(MOVEMENT)])), axis=0)
                highest_objective = info['objective']
            max_future_Q = np.max(Q[(info['objective'],) + new_discrete_state])
            current_Q = Q[(info['objective'],) + new_discrete_state + (action,)]
            new_Q = (1 - LEARNING_RATE) * current_Q + LEARNING_RATE * (reward + DISCOUNT * max_future_Q)
            Q[(info['objective'],) + new_discrete_state + (action,)] = new_Q
        discrete_state = new_discrete_state
        if ep % 10 == 0:
            np.save(f"Q_tables/Q_{ep}.npy", Q)
        if reward < min_reward:
            min_reward = reward
        if reward > max_reward:
            max_reward = reward
        ep_reward += reward
        t += 1
    print("Episode:", ep + 1, "\tep_reward:", "%.2f" % ep_reward, "\tmin_reward:", "%.2f" % min_reward, "\tmax_reward:", "%.2f" % max_reward,
          "\ttarget distance:", "%.2f" % info['target_distance'], "\tepsilon:", "%.2f" % epsilon)

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

env.close()


