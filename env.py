import pandas as pd


class Env(object):
    """
    Environment
    """
    
    def __init__(self, data_path):
        self.ob_size = 1  # only an integer, previous reward
        self.act_size = 3  # -1, 0, +1: short position, idle, long position
        
        self.ds = pd.read_csv(data_path)
        self.day = self.ds['Day'].values
        self.price = self.ds['midPrice'].values
        del self.ds
        
        self.tick = 0
        self.x = [0]  # rewards in history, used as agents' obervation
        self.position = 0

    def reset(self):
        """
        reset the environment, returns the initial observation
        """
        
        self.tick = 0
        self.x = [0]
        self.position = 0
        
        return self.x

    def step(self, action):
        """
        agent take an action
        :param action: the action taken, [-1, 1], representing sell/sold secure
        
        :return: 
        - observation: observation of the next step
        - reward: corresponding reward, defined as ``
        - done: True if entering a new day, else False
        - info: other information
        """

        self.position += action

        if self.position > 5:
            self.position = 5
            action = 0
        elif self.position < -5:
            self.position = -5
            action = 0
        
        reward = action * (self.price[self.tick + 1] - self.price[self.tick])
        
        if self.day[self.tick + 1] != self.day[self.tick]:
            done = True
            self.x = [0]
        else:
            done = False
            self.x.append(reward)
        
        self.tick += 1
        
        return self.x, reward, done, None
        
        
