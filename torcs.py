from keras import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense
from keras.layers import Lambda
from keras.regularizers import l2
import dqn_agent
import eval
import torcs_env

def reward_function(state, done, score, max_score, reward):
    return 1 if done else -1

def get_model(action_size, state_size):

    init = 'glorot_uniform'

    model = Sequential()

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=state_size))

    model.add(Conv2D(24, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(36, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(48, kernel_size=5, activation='relu', strides=(2, 2), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    #model.add(Conv2D(64, kernel_size=3, activation='relu', strides=(1, 1), kernel_initializer=init, kernel_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(units=1164, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=100, kernel_regularizer=l2(0.001)))
    #model.add(Dense(units=50, kernel_regularizer=l2(0.001)))
    #model.add(Dense(units=10, kernel_regularizer=l2(0.001)))
    model.add(Dense(units=action_size, activation='linear'))

    return model


if __name__ == '__main__':
    ## Parameters ##
    resume_train = False

    ##############################

    eval_inst = eval.RLEvaluation(resume_train=resume_train)

    env = torcs_env.TorcsEnvironment(eval_inst=eval_inst)

    # model
    model = get_model(env.action_size, env.state_size)
    agent = dqn_agent.DQNAgent(state_size=env.state_size, action_size=env.action_size, model=model, learning_rate=0.001,
                                      queue_size=50000, batch_size=350, decay_rate=0.95)
    agent.enable_epsilon_greedy(eps_decay=0.999, eps_min=0.1, eps_start=1.0)

    if resume_train == True:
        agent.load_model()

    env.set_agent(agent)

    env.set_reward_func(reward_function)

    env.learn(resume_train =resume_train)
