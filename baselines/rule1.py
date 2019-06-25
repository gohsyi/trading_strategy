import numpy as np


class Model():
    def __init__(self):
        pass

    def step(self, obs):
        obs = np.array(obs)
        actions = np.ones(obs.shape[0])
        actions[obs[:, -1, 1] < 0] = 2
        actions[obs[:, -1, 1] > 0] = 0

        return actions, None
