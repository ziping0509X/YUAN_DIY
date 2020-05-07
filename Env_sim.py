import numpy as np
import random
import sys
import os

TIME_SLOTS = 1
NUM_CHANNELS = 2
NUM_USERS = 3
ATTEMPT_PROB = 0.6
GAMMA = 0.90

class env_sim:
    def __init__(self,num_users,num_channels,attempt_prob):
        self.NUM_USERS = num_users
        self.NUM_CHANNELS = num_channels
        self.REWARD = 1
        self.ATTEMPT_PROB = attempt_prob

        #为action、observation分配存储空间
        self.action_space = np.arange(self.NUM_CHANNELS+1)
        self.users_action = np.zeros([self.NUM_USERS],np.float32)
        self.users_observation = np.zeros([self.NUM_USERS],np.float32)

        def sample(self):
            x = np.random.choice(self.action_space,size=self.NUM_USERS)
            return x

        def step(self,action):

            assert (action.size) == self.NUM_USERS,"action and user should have same dim {}".format(action)
            #assert函数控制AssertionError的值
            #assert 1 == 2, "表达式{}的值不正确".format(2)的输出是：AssertionError: 表达式2的值不正确
            channel_alloc_frequency = np.zeros([self.NUM_USERS+1],np.float32)
            obs = []
            reward = np.zeros([self.NUM_USERS])

























