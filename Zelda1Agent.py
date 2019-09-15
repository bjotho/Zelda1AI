from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT
import numpy as np
from random import randint


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


def get_discrete_state(state):
    x_i = state[0] // 16
    y_i = (state[1] - 61) // 16
    if x_i == 15:
        x_i -= 1
    if y_i == 10:
        y_i -= 1
    return tuple((x_i, y_i))


def model_output(input):
    global model
    return np.argmax(model.predict(np.array([input]))[0])


def build_model():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    import keras

    global model
    model = Sequential()
    i = 0
    # Input dim: (240,256)
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(240,256,3)))
    print("Input", model.input_shape)
    # Input dim: (238,254)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    # Input dim: (236,252)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Input dim: (118,126)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    # Input dim: (116,124)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    # Input dim: (114,122)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Input dim: (57,61)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Conv2D(32, kernel_size=(4,4), activation='relu'))
    # Input dim: (54,58)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    # Input dim: (52,56)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Input dim: (26,28)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    # Input dim: (24,26)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    # Input dim: (22,24)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Input dim: (11,12)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Conv2D(32, kernel_size=(2,3), activation='relu'))
    # Input dim: (10,10)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    # Input dim: (8,8)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Input dim: (4,4)
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Dropout(0.25))
    # Input dim: (4,4))
    print("Layer", i, model.output_shape)
    i += 1
    model.add(Flatten())
    # Output dim: 512
    print("Layer", i, model.output_shape)
    i += 1
    # model.add(Dense(512, activation='relu'))
    # print("Layer", i, model.output_shape)
    # i += 1
    # model.add((Dropout(0.5)))
    # print("Layer", i, model.output_shape)
    # model.add(Dense(env.action_space.n, activation='softmax'))
    # print("Output", model.output_shape)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

env = gym_zelda_1.make('Zelda1-v0')
env = JoypadSpace(env, MOVEMENT)
# model = None
# build_model()
# The area where Link's _x_pixel can be is approximately 240*160 pixels (x:0-240, y:61-221).
# If we divide these dimensions by 16, we get a (15, 10) matrix For each position Link can be in,
# he can perform 20 distinct actions. Therefore, the Q matrix will have the dimensions [10,15,20].
Q = np.random.uniform(low=-15, high=15, size=([15,10,7]))
# print(Q)

get_sword = 0
state = env.reset()

if get_sword:
    # Go get sword for agent
    a = []
    for i in range(45):
        a.append(8)
    for i in range(165):
        a.append(12)
    for i in range(10):
        a.append(4)
    for i in range(200):
        a.append(16)

    # Start x and y pos: (120, 141)
    for step in range(len(a)+100):
        if step < len(a)-1:
            state, reward, done, info = env.step(a[step])
        else:
            if info['y_pos'] < 141:
                state, reward, done, info = env.step(16)
            elif info['x_pos'] < 120:
                state, reward, done, info = env.step(4)
            else:
                state, reward, done, info = env.step(0)
        env.render()
    print("Agent taking over")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
for ep in range(EPISODES):
    min_reward = 15
    max_reward = -15
    t = 0
    state = env.reset()
    done = False
    state, reward, done, info = env.step(0)
    discrete_state = get_discrete_state((info['x_pos'], info['y_pos']))
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmin(Q[get_discrete_state((info['x_pos'], info['y_pos']))])
        new_state, reward, done, info = env.step(action)
        new_discrete_state = get_discrete_state((info['x_pos'], info['y_pos']))
        if ep % 1 == 0:
            env.render()
        if not done:
            max_future_Q = np.max(Q[new_discrete_state[0]][new_discrete_state[1]])
            current_Q = Q[discrete_state[0]][discrete_state[1]][action]
            new_Q = (1 - LEARNING_RATE) * current_Q + LEARNING_RATE * (reward + DISCOUNT * max_future_Q)
            Q[discrete_state[0]][discrete_state[1]][action] = new_Q
        discrete_state = new_discrete_state
        if reward < min_reward:
            min_reward = reward
        if reward > max_reward:
            max_reward = reward
        t += 1
    if END_EPSILON_DECAYING >= t >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    print("Episode:", ep+1, "\tmin_reward:", "%.2f" % min_reward, "\tmax_reward:", "%.2f" % max_reward, "\ttarget distance:", "%.2f" % info['target_distance'], "\tepsilon:", "%.2f" % epsilon)
env.close()

for i in Q:
    for j in i:
        print(j)

'''
timesteps = 10000
# eta = .628
# gamma = .9
# # reward_list = []
# total_reward = 0
# observation = model_output(state)
# for step in range(timesteps):
#     action = np.argmax(Q[observation] + np.random.randn(1, env.action_space.n) * (1./(step+1)))
#     state, reward, done, info = env.step(action)
#     next_observation = model_output(state)
#     Q[observation][action] = Q[observation][action] + eta * (reward + gamma * np.max(Q[next_observation]) - Q[observation][action])
#     total_reward += reward
#     observation = next_observation
#     if reward > 0.01:
#         print("Reward:", "%.2f" % total_reward - reward, "-->", "%.2f" % total_reward)
#     env.render()
# env.close()
# 
# print("Reward sum for all timesteps:", total_reward/timesteps)
# print("Final Q-table")
# print(Q)
# print("Reward list:", reward_list)



# for step in range(timesteps):
#     action = np.argmax(model.predict(np.array([state]))[0])
#     state, reward, done, info = env.step(action)
#     if reward > 0:
#         print("action:", action, "reward:", reward, "step:", step)
#         y = [0 for i in range(env.action_space.n)]
#         y[action] = 1
#         model.fit(np.array([state]), np.array([y]), epochs=1, verbose=0)
#     else:
#         if reward < 0:
#             print("action:", action, "penalty:", reward, "step:", step)
#         y = [0 for i in range(env.action_space.n)]
#         i = randint(0,env.action_space.n-1)
#         if i == action:
#             i = (i+randint(1,env.action_space.n-2)) % env.action_space.n-1
#         y[i] = 1
#         model.fit(np.array([state]), np.array([y]), epochs=1, verbose=0)
#     if step % int(timesteps/10) == 0:
#         print(int(100*step/timesteps), "%")
#     env.render()
# 
# env.close()
# 
# print(model.summary())
'''