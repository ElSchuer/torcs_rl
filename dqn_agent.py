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

    def __init__(self, state_size, action_size, model, decay_rate=0.95, learning_rate=0.001, model_name='model.h5', batch_size=100, queue_size=10000):
        self.model_name = model_name

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        self.batch_size = batch_size
        self.data_batch = deque(maxlen=queue_size)

        self.model = model
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_model(self):
        self.model.save(self.model_name)

    def load_model(self):
        self.model.load_weights(self.model_name)

    def act(self, state):
        return np.argmax(self.model.predict(state))

    def remember(self, state, next_state, reward, action, done):
        self.data_batch.append([state, next_state, reward, action, done])

    def train(self):
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
                q_future = self.model.predict(next_state)[0]
                target[0][action] = reward + self.decay_rate * np.amax(q_future)

            states.append(state[0])
            targets.append(target[0])

        history = self.model.fit(np.array(states), np.array(targets), verbose=0, epochs=1)

        return history

class EGreegyDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, model, decay_rate=0.95, learning_rate=0.001, model_name='model.h5', batch_size=100, queue_size=10000,
                 eps_start=1.0, eps_min=0.01, eps_decay=0.999):
        
        super(EGreegyDQNAgent, self).__init__(state_size, action_size=action_size, model=model, learning_rate=learning_rate, model_name=model_name,
                                                batch_size=batch_size, queue_size=queue_size, decay_rate=decay_rate)

        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def act(self, state):

        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)

        return np.argmax(self.model.predict(state))

    def decay_epsilon(self):
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def train(self):
        self.decay_epsilon()
        return self.train_batch()


class TargetNetworkDQNNAgent(EGreegyDQNAgent):

    def __init__(self, state_size, action_size, model, decay_rate=0.95, learning_rate=0.001, model_name='model.h5', batch_size=100, queue_size=10000,
                 eps_start=1.0, eps_min=0.01, eps_decay=0.999, update_steps = 5000):

        super().__init__(self, action_size=action_size, model=model, decay_rate=decay_rate,
                         batch_size=batch_size, model_name=model_name, learning_rate=learning_rate,
                         queue_size=queue_size, eps_start=eps_start, eps_min=eps_min, eps_decay=eps_decay)

        self.target_model = self.model
        self.update_steps = update_steps
        self.step = 0

    def update_target_model(self):
        if self.step % self.update_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
            print("Update target model")

        self.step += 1

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
                q_future = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.decay_rate * np.amax(q_future)

            states.append(state[0])
            targets.append(target[0])

        history = self.model.fit(np.array(states), np.array(targets), verbose=0, epochs=1)

        return history

    def remember(self, state, next_state, reward, action, done):
        self.update_target_model()

        super().remember(state, next_state, reward, action, done)


class DoubleDQNAgent(TargetNetworkDQNNAgent):

    def __init__(self, state_size, action_size, model, decay_rate=0.95, learning_rate=0.001, model_name='model.h5', batch_size=100, queue_size=10000,
                 eps_start=1.0, eps_min=0.01, eps_decay=0.999, update_steps = 5000):

        super().__init__(self, state_size=state_size, action_size=action_size, model=model, decay_rate=decay_rate,
                         batch_size=batch_size, model_name=model_name, learning_rate=learning_rate,
                         queue_size=queue_size, eps_start=eps_start, eps_min=eps_min, eps_decay=eps_decay, update_steps=update_steps)

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
                q_target = self.target_model.predict(next_state)[0]
                q_next = self.model.predict(next_state)[0]
                next_action = np.argmax(q_next)
                target[0][action] = reward + self.decay_rate * q_target[next_action]

            states.append(state[0])
            targets.append(target[0])

        history = self.model.fit(np.array(states), np.array(targets), verbose=0, epochs=1)

        return history

class DuelingDDQNAgent(TargetNetworkDQNNAgent):

    def __init__(self, state_size, action_size, model, decay_rate=0.95, learning_rate=0.001, model_name='model.h5', batch_size=100, queue_size=10000,
                 eps_start=1.0, eps_min=0.01, eps_decay=0.999, update_steps = 5000):

        model = self.get_dueling_model(model, action_size)

        super().__init__(self, action_size=action_size, model=model, decay_rate=decay_rate,
                         batch_size=batch_size, model_name=model_name, learning_rate=learning_rate,
                         queue_size=queue_size, eps_start=eps_start, eps_min=eps_min, eps_decay=eps_decay, update_steps=update_steps)



    def get_dueling_model(self, model, action_size):

        last_layer = model.layers[-2]

        y = Dense(action_size + 1, activation='linear')(last_layer.output)
        outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True),
                             output_shape=(action_size,))(y)

        model = Model(inputs = model.input, outputs = outputlayer)

        model.summary()

        return model
