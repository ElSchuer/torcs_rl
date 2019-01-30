from keras import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras.layers import Lambda
from keras.regularizers import l2
import dqn_agent
import eval
import torcs_env



def reward_function(state, done, score, max_score, reward):
    return reward

def get_model(action_size, state_size):

    dense_keep_prob = 0.8
    init = 'glorot_uniform'

    model = Sequential()

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=state_size))

    model.add(Conv2D(24, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(36, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(48, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(units=1164, kernel_regularizer=l2(0.001)))
    model.add(Dropout(rate=dense_keep_prob))

    model.add(Dense(units=100, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=50, kernel_regularizer=l2(0.001)))
    #model.add(Dense(units=10, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=action_size, activation='linear'))

    return model


eval_inst = eval.RLEvaluation()
env = torcs_env.TorcsEnvironment(eval_inst=eval_inst)

# model
model = get_model(env.action_size, env.state_size)

agent = dqn_agent.DuelingDDQNAgent(state_size=env.state_size, action_size=env.action_size, model=model, learning_rate=0.001,
                                 queue_size=50000, batch_size=256, eps_decay=0.999, eps_min=0.05, decay_rate=0.95, update_steps=10000)

env.set_agent(agent)
env.set_reward_func(reward_function)

env.learn()