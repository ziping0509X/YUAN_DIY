import tensorflow as tf
import numpy as np
from collections import deque
import random

FINAL_EPSILON = 0.0
INITIAL_EPSILON = 0.8
GAMMA = 0.8
OBSERVE = 300
EXPLORE = 100000
REPLAY_MEMORY = 400
BATCH_SIZE = 256

class DRQN:
    def __init__(self, input_size, time_step, hidden_size, action_size,name='DRQN',learning_rate = 10 ** -5):
        self.inputsize = input_size
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.epsilon = INITIAL_EPSILON
        self.timeStep = 0
        self.recording = EXPLORE

        self.replayMemory = ReplayMemory(Max_size = REPLAY_MEMORY)

        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32, [None,  input_size])

            self.lstm = tf.contrib.rnn.BasicLSTMCell(input_size)
            self.lstm_out, self.state = tf.nn.dynamic_rnn(self.lstm, self.input, dtype=tf.float32)

            self.reduced_out = self.lstm_out[:, -1, :]
            self.reduced_out = tf.reshape(self.reduced_out, shape=[-1, hidden_size])

            self.w2 = tf.Variable(tf.random_uniform([hidden_size, hidden_size]))
            self.b2 = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
            self.h2 = tf.matmul(self.reduced_out, self.w2) + self.b2
            self.h2 = tf.nn.relu(self.h2)
            self.h2 = tf.contrib.layers.layer_norm(self.h2)

            self.w3 = tf.Variable(tf.random_uniform([hidden_size, action_size]))
            self.b3 = tf.Variable(tf.constant(0.1, shape=[action_size]))
            self.output = tf.matmul(self.h2, self.w3) + self.b3

            self.Q = self.output

            self.actionInput = tf.placeholder("float", [None, self.action_size])
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            Q_action = tf.reduce_sum(tf.multiply(self.Q, self.actionInput), reduction_indices=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - Q_action))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.session = tf.InteractiveSession()
            self.session.run(tf.global_variables_initializer())

    def setInitState(self, observation):
        self.currentState = observation

    def getAction(self):
        QValue = self.Q.eval(feed_dict={self.input: [self.currentState]})
        action = np.zeros(self.action_size)
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action_size)
            action[action_index] = 1
        else:
            action_index = np.argmax(QValue)
            action[action_index] = 1

        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            self.recording = self.recording - 1

        return action, self.recording

    def setPerception(self, nextObservation, action, reward):
        #整个DRQN网络的入口地址
        loss = 0
        newState = nextObservation
        self.replayMemory.add((self.currentState, action, reward, newState))

        if self.timeStep > OBSERVE:
            loss = self.trainQNetwork()
        self.currentState = newState
        self.timeStep += 1
        return loss

    def trainQNetwork(self):

        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        y_batch = []
        QValue_batch = self.Q.eval(feed_dict={self.input: nextState_batch})
        for i in range(0, BATCH_SIZE):
            y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        _, self.loss = self.session.run([self.opt, self.loss], feed_dict={
            self.targetQs_: y_batch,
            self.actionInput: action_batch,
            self.input: state_batch
        })
        return self.loss


class ReplayMemory:
    def __init__(self,Max_size):
        self.replaymemory = deque(maxlen = Max_size)

    def add(self,currentState, action, reward, newState):
        self.replaymemory.append((currentState, action, reward, newState))

    def sample(self, batch_size, step_size):
        idx = np.random.choice(np.arange(len(self.replaymemory) - step_size),
                               size=batch_size, replace=False)

        res = []

        for i in idx:
            temp_buffer = []
            for j in range(step_size):
                temp_buffer.append(self.replaymemory[i+j])

            res.append(temp_buffer)

        return res







