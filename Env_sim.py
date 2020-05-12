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
            #这里传入的action是一个动作数组，这个动作数组就是当前所有智能体选择的信道结果
            #传入的当前slot的action，穿出的是observation结果

            assert (action.size) == self.NUM_USERS,"action and user should have same dim {}".format(action)
            #assert函数控制AssertionError的值
            #assert 1 == 2, "表达式{}的值不正确".format(2)的输出是：AssertionError: 表达式2的值不正确
            channel_alloc_frequency = np.zeros([self.NUM_USERS+1],np.float32)
            obs = [] #
            reward = np.zeros([self.NUM_USERS])

            j=0
            for each in action:
                prob = random.uniform(0,1)
                if prob <= self.ATTEMPT_PROB:
                    self.users_action[j] = each
                    channel_alloc_frequency[each]+=1
                j=j+1
            #这里出现了三个数组：users_action\外部传入的action\channel_alloc_frequency

            for i in range(1,len(channel_alloc_frequency)):
                if channel_alloc_frequency[i] > 1:
                    channel_alloc_frequency[i] = 0

                for i in range(len(action)):
                    self.users_observation [i] = channel_alloc_frequency[self.users_action[i]]
                    if self.users_action[i] == 0:
                        self.users_observation[i] = 0
                    if self.users_observation[i] == 1
                        reward[i] = 1
                    obs.append((self.users_observation[i],reward[i]))#存储的是一组（A,B）坐标
            residual_channel_capacity = channel_alloc_frequency[1:]
            residual_channel_capacity = 1-residual_channel_capacity #这里存有疑问，1-residual_channel_capacity是什么意思
            obs.append(residual_channel_capacity)

            return obs






























