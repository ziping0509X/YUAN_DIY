from Env_sim import env_sim
from Qnetwork import DQN,Memory
from collections import deque
import numpy as np
import sys
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time

TIME_SLOTS = 100000
NUM_CHANNELS = 2
NUM_USERS = 3
ATTEMPT_PROB = 1

