from gym_torcs import TorcsEnv
import numpy as np
import time
from environment import Environment
import os
import cv2

class TorcsEnvironment(Environment):
    def __init__(self, eval_inst, seed = 10, max_episodes = 100000, train_steps=80):
        super().__init__(max_episodes=max_episodes, seed=seed)

        self.env = TorcsEnv()

        self.eval_inst = eval_inst

        self.train_steps = train_steps

        self.state_size = (96, 128, 1)
        print('state size', self.state_size)

        self.action_size = self.env.action_space.n
        print('action space', self.env.action_space)
        print('action size', self.env.action_space.n)
        self.old_timestamp = None

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
        state = np.flipud(np.resize(ob[0], self.state_size))
        return state[None, :, :, :]

    def log_timeout_warning(self):
        f = open(self.eval_inst.logfile, 'a')       
        f.write('[WARNING]; Timeout: Time consumed by "torcs_env.learn()" > 1 sec \n')
        f.close()

    def learn(self, resume_train):
        episode_start = 0
        if resume_train is True:
            episode_start=self.eval_inst.episodes[-1]+1

        for e in range(episode_start, self.max_episodes):
            
            if np.mod(e, 3) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                state = self.convert_oberservation_to_state(self.env.reset(relaunch=True))
            else:
                state = self.convert_oberservation_to_state(self.env.reset())

            total_reward = 0

            s = 0

            loss_values = []

            while True:
                
                #if self.old_timestamp is not None and time.time()-self.old_timestamp > 1.0:
                    #self.log_timeout_warning()
                
                s += 1
               
                action = self.agent.act(state)
                
                next_state, reward, done, info = self.env.step(action)
                
                next_state = self.convert_oberservation_to_state(next_state)

                img = next_state[0,:,:,0]
                cv2.imshow('image',img)
                cv2.waitKey(1)

                reward = self.get_reward(state, done, s, 0, reward)
                print(reward)
                total_reward += reward
                
                self.agent.remember(state, next_state, reward, action, done)
                
                hist = self.train_agent(s)

                if hist is not None:
                    loss_values.append(hist.history.get("loss")[0])
                
                state = next_state
                
                if done:
                    self.agent.save_model()

                    #print("Episode {}, score {}, loss {:.2}, eps {:.4}, reward {}".format(e, s,
                      #     np.mean(loss_values), self.agent.eps, total_reward))

                    self.eval_inst.visualize_data(e, np.mean(loss_values), s)

                    break
