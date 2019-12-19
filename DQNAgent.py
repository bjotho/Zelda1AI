from keras import Input
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import keras
import tensorflow as tf

from collections import deque
import numpy as np
import time
import random

from gym_zelda_1.actions import MOVEMENT

# Ignore some warnings. Ignorance is bliss.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if tf.__version__ == '2.0.0':
    # Disable eager execution for compatability with tf.compat.v1.summary.FileWriter()
    tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.disable_v2_behaviour()
    tf.compat.v1.experimental.output_all_intermediates(True)


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
    def __init__(self, model=None):

        self.DISCOUNT = 0.99
        self.REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training.
        self.MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training.
        self.MINIBATCH_SIZE = 64  # How many steps (samples) to use for training.
        self.UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes).
        self.CONV_FILTERS = 4
        self.MODEL_NAME = "5x" + str(self.CONV_FILTERS)
        self.MIN_REWARD = -2000  # For model save.
        self.MEMORY_FRACTION = 0.20

        self.dim = (240, 256)  # Dimensions of the state image.
        self.left_crop = 0
        self.top_crop = 0
        self.right_crop = self.dim[1]
        self.bottom_crop = self.dim[0]

        # Main model. This gets trained every step.
        self.model = self.create_model(model)
        self.model.summary()

        # Target model. This is what we .predict against every step.
        self.target_model = self.create_model(model)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}")

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self, model):
        if model is not None:
            # Load JSON and create model
            json_file = open(f"models/{model}.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # Load weights into new model
            loaded_model.load_weights(f"models/{model}.h5")
            loaded_model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
            print("Loaded model:", model)
            return loaded_model

        # With entire image: input_shape = (240, 256, 3)
        model = Sequential()
        model.add(Conv2D(self.CONV_FILTERS, (3, 3), input_shape=(self.dim[0] - (self.top_crop + (self.dim[0] - self.bottom_crop)), self.dim[1] - (self.left_crop + (self.dim[1] - self.right_crop)), 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.1))
        model.add(Conv2D(self.CONV_FILTERS, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.1))
        model.add(Conv2D(self.CONV_FILTERS, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.1))
        model.add(Conv2D(self.CONV_FILTERS, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.1))
        model.add(Conv2D(self.CONV_FILTERS, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.1))
        model.add(Flatten()) # this converts our 3D feature map to 1D feature vector
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dense(32))
        model.add(Activation("relu"))
        model.add(Dense(16))
        model.add(Activation("relu"))
        print(model.output_shape)
        model.add(Dense(len(MOVEMENT))) # len(MOVEMENT) = number of possible actions
        model.add(Activation("linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we only use part of the equation here
            if not done:
                max_future_Q = np.max(future_qs_list[index])
                new_Q = reward + self.DISCOUNT * max_future_Q
            else:
                new_Q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_Q

            # And append to our training data
            X.append(current_state)
            Y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(Y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0



    # training method used by the aleph_star algorithm
    def trainit(self, N, states, mqs, lr, batchsize, priorities):
        h = np.sort(priorities)[int(np.floor(0.9 * len(priorities)))]
        _p = [np.max([h, p]) for p in priorities]
        norm_p = [float(i)/np.sum(_p) for i in _p]
        # tll = 0.0
        for j in range(N):
            print(j+1, "of", N)
            self.batchit(states, mqs, batchsize, norm_p, j+1, N)

    # batch training method used by the aleph_star algorithm
    def batchit(self, states, qs, batchsize, p, round, last_round):
        NACT = len(qs[0])
        X = np.zeros(shape=(batchsize,) + states[0].shape)
        Y = np.zeros(shape=(batchsize, NACT))
        bix = []
        for i in range(batchsize):
            bix.append(np.random.choice(len(p), p=p))
        for bnum, ix in enumerate(bix):
            Y[bnum] = qs[ix]
            X[bnum] = states[ix] / 255
        self.model.fit(X, Y, batch_size=batchsize, verbose=0, shuffle=False, callbacks=[self.tensorboard] if round >= last_round else None)