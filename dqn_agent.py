import numpy as np
import random
from collections import deque
from keras.optimizers import Adam
from keras.layers import Lambda
from keras.layers import Dense
from keras import backend as K
from keras.models import Model
import tensorflow as tf


class DQNAgent:

    def __init__(self, state_size, action_size, model, decay_rate=0.95, learning_rate=0.001, model_name='model.h5',
                 batch_size=100, queue_size=10000, loss='mse'):
        self.model_name = model_name

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        self.batch_size = batch_size
        self.data_batch = deque(maxlen=queue_size)

        self.model = model
        self.loss = loss
        self.compile_model(loss)

        # Epsilon Greegy Extension Parameters
        self.eps_greegy_enabled = False
        self.eps = 1
        self.eps_min = 0.01
        self.eps_decay = 0.995

        # Target Network Extension Parameters
        self.target_network_enabled = False
        self.target_model = self.model
        self.update_steps = 5000
        self.step = 0

        # Double DQN Extension Parameter
        self.double_dqn_enabled = False

        # Dueling DQN Extension Parameter
        self.dueling_dqn_enabled = False
        self.dueling_type = 'naive'

    def compile_model(self, loss):
        if loss == 'huber':
            self.model.compile(loss=self.huber_loss, optimizer=Adam(lr=self.learning_rate))
        else:
            self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def enable_epsilon_greedy(self, eps_min, eps_start, eps_decay):
        self.eps_greegy_enabled = True
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def decay_epsilon(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def enable_target_network(self, update_steps):
        self.target_network_enabled = True
        self.update_steps = update_steps

    def update_target_model(self):
        if self.step % self.update_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
            print("Update target model")

        self.step += 1

    def enable_double_dqn(self):
        self.double_dqn_enabled = True

    def enable_dueling_dqn(self, dueling_type):
        self.dueling_dqn_enabled = True
        self.dueling_type = dueling_type

        self.model = self.get_dueling_model(self.model, self.action_size)
        self.compile_model(self.loss)

    def get_dueling_model(self, model, action_size):
        last_layer = model.layers[-2]

        y = Dense(action_size + 1, activation='linear')(last_layer.output)

        if self.dueling_type == 'naive':
            output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:],
                            output_shape=(action_size,))(y)
        elif self.dueling_type == 'mean':
            output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True),
                            output_shape=(action_size,))(y)
        elif self.dueling_type == 'max':
            output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True),
                            output_shape=(action_size,))(y)

        model = Model(inputs=model.input, outputs=output)

        model.summary()

        return model

    def save_model(self):
        self.model.save(self.model_name)

    def load_model(self):
        self.model.load_weights(self.model_name)

    def act(self, state):
        if self.eps_greegy_enabled and np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state, next_state, reward, action, done):

        if self.target_network_enabled:
            self.update_target_model()

        self.data_batch.append([state, next_state, reward, action, done])

    def train(self):

        if self.eps_greegy_enabled:
            self.decay_epsilon()

        return self.train_batch()

    def huber_loss(self, y_true, y_pred, clipping_delta=1.0):
        ## hueber loss function

        err = y_true - y_pred

        cond = K.abs(err) < clipping_delta

        squared_loss = 0.5 * K.square(err)
        quadratic_loss = 0.5 * K.square(clipping_delta) + clipping_delta * (K.abs(err) - clipping_delta)

        loss = tf.where(cond, squared_loss, quadratic_loss)

        return K.mean(loss)

    def train_batch(self):

        tmp_batch = random.sample(self.data_batch, self.batch_size)

        states = []
        targets = []

        for state, next_state, reward, action, done in tmp_batch:

            # q(a,s)
            target = self.model.predict(state)

            if done:
                target[0][action] = reward
            else:
                # q(a', s')
                if self.target_network_enabled:
                    q_target = self.target_model.predict(next_state)[0]
                else:
                    q_target = self.model.predict(next_state)[0]

                if self.double_dqn_enabled:
                    q_next = self.model.predict(next_state)[0]
                    next_action = np.argmax(q_next)
                    target[0][action] = reward + self.decay_rate * q_target[next_action]
                else:
                    target[0][action] = reward + self.decay_rate * np.amax(q_target)

            states.append(state[0])
            targets.append(target[0])

        history = self.model.fit(np.array(states), np.array(targets), verbose=0, epochs=1)

        return history
