from power_control import GameState
from DRQN import DRQN
import numpy as np
import matplotlib.pyplot as plt

P_1 = [round(0.1 * i / 2.0, 2) for i in range(1, 9)]
P_2 = [round(0.1 * i / 2.0, 2) for i in range(1, 9)]
actions = len(P_2)

Loss = []
Success = []
Fre = []

noise = 3
num_sensor = 10  # N
policy = 2  # choose power change policy for PU, it should be 1(Multi-step) or 2(Single step)

DRQN = DRQN(input_size=10, time_step=1, hidden_size=256, action_size = actions,name='DRQN',learning_rate=10**-5)

com = GameState(P_1, P_2, noise, num_sensor)
terminal = True
recording = 100000

while (recording > 0):
    # initialization
    if (terminal == True):
        com.ini()
        observation0, reward0, terminal = com.frame_step(np.zeros(actions), policy, False)
        DRQN.setInitState(observation = observation0)

    # train
    action, recording = DRQN.getAction()
    nextObservation, reward, terminal = com.frame_step(action, policy, True)
    loss = DRQN.setPerception(nextObservation, action, reward)

# plt.plot(Loss)
# # plt.show()
# #
# # plt.plot(Success)
# # plt.show()
# #
# # plt.plot(Fre)
# # plt.show()