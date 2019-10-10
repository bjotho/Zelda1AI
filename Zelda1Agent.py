from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf

from collections import deque
import numpy as np
import time
import random


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided, we train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, _, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):

        # Main model. This gets trained every step.
        self.model = self.create_model()

        # Target model. This is what we .predict against every step.
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        # model.add(Conv2D(256, (3, 3), input_shape=(176, 256, 3)))
        model.add(Conv2D(16, (3, 3), input_shape=(dim[0] - (top_crop + (dim[0] - bottom_crop)), dim[1] - (left_crop + (dim[1] - right_crop)), 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(env.action_space.n, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
            if not done:
                max_future_Q = np.max(future_qs_list[index])
                new_Q = reward + DISCOUNT * max_future_Q
            else:
                new_Q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_Q

            X.append(current_state)
            Y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Updating to determine if we want to update target_model yet.
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


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


def get_discrete_state(state):
    x_i = state[0] // 16
    y_i = (state[1] - 61) // 16
    return tuple((x_i, y_i))


def discrete_observation(state, left_crop=0, top_crop=0, right_crop=None, bottom_crop=None):
    """Basically crops the input image (state) to the specified pixels"""
    observation = []
    for i in state[top_crop:(bottom_crop if bottom_crop < dim[0] else None)]:
        observation.append([])
        for j in i[left_crop:(right_crop if right_crop < dim[1] else None)]:
            observation[-1].append([])
            for k in j:
                observation[-1][-1].append(k)

    return np.array(observation)


env = gym_zelda_1.make('Zelda1-v0')
env = JoypadSpace(env, MOVEMENT)
# The area where Link can be is approximately 255*175 pixels (x:0-255, y:64-239).
# If we divide these dimensions by 16, we get a (16, 11) matrix. This matrix will represent each discrete position Link can be in,
# and for each of these discrete positions, he can perform len(MOVEMENT) distinct actions. Therefore, the Q matrix will have the dimensions [11,16,len(MOVEMENT)].
# Q = np.random.uniform(low=-15, high=15, size=([1, 16, 11, len(MOVEMENT)]))
# print(Q.shape)

start_in_level_1 = 0
state = env.reset()

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000 # How many last steps to keep for model training.
MIN_REPLAY_MEMORY_SIZE = 1_000 # Minimum number of steps in a memory to start training.
MINIBATCH_SIZE = 1 # How many steps (samples) to use for training.
UPDATE_TARGET_EVERY = 5 # Terminal states (end of episodes).
MODEL_NAME = "2x256"
MIN_REWARD = -1000 # For model save.
MEMORY_FRACTION = 0.20

# Environment settings.
EPISODES = 100

# Exploration settings.
epsilon = 1 # Not a constant, will decay.
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
            action = np.argmax(agent.get_qs(observation))
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
