from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf

from collections import deque
import numpy as np
import time
import random

from gym_zelda_1.actions import MOVEMENT

# Ignore some warnings. Ignorance is bliss.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable eager execution for compatability with tf.compat.v1.summary.FileWriter()
tf.compat.v1.disable_eager_execution()

if(tf.__version__ == '2.0.0'):
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.experimental.output_all_intermediates(True)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000 # How many last steps to keep for model training.
MIN_REPLAY_MEMORY_SIZE = 1_000 # Minimum number of steps in a memory to start training.
MINIBATCH_SIZE = 1 # How many steps (samples) to use for training.
UPDATE_TARGET_EVERY = 5 # Terminal states (end of episodes).
MODEL_NAME = "2x256"
MIN_REWARD = -1000 # For model save.
MEMORY_FRACTION = 0.20


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

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

        self.dim = (240, 256)  # Dimensions of the state image.
        self.left_crop = 0
        self.top_crop = 0
        self.right_crop = self.dim[1]
        self.bottom_crop = self.dim[0]

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
        # model.add(Conv2D(256, (3, 3), input_shape=(240, 256, 3)))
        model.add(Conv2D(1, (3, 3), input_shape=(self.dim[0] - (self.top_crop + (self.dim[0] - self.bottom_crop)), self.dim[1] - (self.left_crop + (self.dim[1] - self.right_crop)), 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(len(MOVEMENT), activation="linear"))
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
