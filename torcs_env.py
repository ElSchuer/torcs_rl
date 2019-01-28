from gym_torcs import TorcsEnv
import numpy as np
import time
from environment import Environment
import os

class TorcsEnvironment(Environment):
    def __init__(self, eval_inst, seed = 10, train_steps=10):
        super().__init__(self, seed=seed)

        self.env = TorcsEnv()

        self.eval_inst = eval_inst

        self.train_steps = train_steps

        self.state_size = (128, 128, 1)
        print('state size', self.state_size)

        self.action_size = self.env.action_space.n
        print('action space', self.env.action_space)
        print('action size', self.env.action_space.n)

    def step(self, action):
        return self.env.step(action)

    def train_agent(self, step):

        if len(self.agent.data_batch) >= self.agent.batch_size and step % self.train_steps == 0:
            time.sleep(0.1)
            os.system("xte 'key F1'")
            history = self.agent.train()
            time.sleep(0.1)
            os.system("xte 'key F1'")
            return history

    def convert_oberservation_to_state(self, ob):
        state = np.resize(ob[0], self.state_size)
        return state[None, :, :, :]

    def learn(self):
        for e in range(10000):
            if np.mod(e, 3) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                state = self.convert_oberservation_to_state(self.env.reset(relaunch=True))
            else:
                state = self.convert_oberservation_to_state(self.env.reset())

            total_reward = 0

            s = 0

            hist = None

            while True:
                s += 1
                action = self.agent.act(state)

                next_state, reward, done, info = self.env.step(action)
                next_state = self.convert_oberservation_to_state(next_state)

                reward = self.get_reward(state, done, s, 0, reward)
                total_reward += reward

                self.agent.remember(state, next_state, reward, action, done)

                hist = self.train_agent(s)

                state = next_state

                if done:
                    self.agent.save_model()
                    if hist is not None:
                        print("Episode {}, score {}, loss {:.2}, eps {:.4}, reward {}".format(e, s,
                            hist.history.get("loss")[0], self.agent.eps, total_reward))

                    self.eval_inst.visualize_data(e, hist.history.get("loss")[0] if hist is not None else 0, s)

                    break