import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time


class TorcsEnv:
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True


    def __init__(self):
    
        self.initial_run = True

       
        os.system('pkill torcs')
        time.sleep(0.5)
       
        os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

        ######CHANGED FOR DISCRETE VALUES######
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.action_space = spaces.Discrete(21)
        print(self.action_space)

        high = np.array([255])
        low = np.array([0])
        self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):

        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
       
        target_speed = self.default_speed
        if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
            client.R.d['accel'] += .01
        else:
            client.R.d['accel'] -= .01

        if client.R.d['accel'] > 0.2:
            client.R.d['accel'] = 0.2

        if client.S.d['speedX'] < 10:
            client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
        if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
            (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
            action_torcs['accel'] -= .2
        
        action_torcs['gear'] = 1

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()

        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        progress = sp*np.cos(obs['angle'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement #########################
        episode_terminate = False
        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = - 1
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
      
        os.system('pkill torcs')
        time.sleep(0.5)
       
        os.system('torcs -nofuel -nodamage -nolaptime -vision &')
       
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        #######CHANED FROM u[0] to u###########
        torcs_action = {'steer': u}

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        
        image_vec = obs_image_vec
    
        return np.array(image_vec, dtype=np.uint8)

    def make_observaton(self, raw_obs):
	
        names = ['img']
        Observation = col.namedtuple('Observaion', names)

        # Get RGB from observation

        image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[0]])

        return Observation(img=image_rgb)
