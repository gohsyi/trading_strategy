import numpy as np


class Model():
    def __init__(self, act_size):
        self.act_size = act_size

    def step(self, obs):
        return np.random.randint(0, self.act_size, size=len(obs)), None
