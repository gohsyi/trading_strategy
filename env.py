from common import args

import pandas as pd
import xgboost as xgb


class Env(object):
    """
    Environment
    """
    
    def __init__(self, data_path):
        self.ob_size = 2  # observation, (price, predicted price 10 time steps after)
        self.act_size = 3  # -1, 0, +1: short position, idle, long position

        dataset = pd.read_csv(data_path)

        indicators = dataset.columns.values[:108].tolist()
        market_stat = ['midPrice', 'LastPrice', 'Volume', 'LastVolume', 'Turnover', 'LastTurnover',
                       'OpenInterest', 'UpperLimitPrice', 'LowerLimitPrice', 'am_pm', 'UpdateMinute']
        feature = indicators + market_stat
        
        data = dataset[feature]
        label = dataset['label']

        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(args.xgb_path)  # load model

        self.pred = bst.predict(xgb.DMatrix(data, label=label))
        self.day = dataset['Day'].values
        self.price = dataset['midPrice'].values

        self.tick = 0
        self.x = [self.get_observation()]
        self.position = 0

    def reset(self):
        """
        reset the environment, returns the initial observation
        """
        
        self.tick = 0
        self.x = [self.get_observation()]
        self.position = 0
        
        return self.x

    def get_observation(self):
        return (self.price[self.tick], self.pred[self.tick])

    def step(self, raw_action):
        """
        agent take an action
        :param raw_action: the action taken, {0, 1, 2}, representing short/idle/long
        
        :return: 
        - observation: observation of the next step
        - reward: corresponding reward, defined as ``
        - done: True if entering a new day, else False
        - info: other information
        """

        action = raw_action - 1
        self.position += action

        if self.position > args.max_position:
            self.position = args.max_position
            action = 0
        elif self.position < -args.max_position:
            self.position = -args.max_position
            action = 0

        # next timestep
        self.tick += 1
        reward = action * (self.price[self.tick] - self.price[self.tick - 1])

        if self.day[self.tick] != self.day[self.tick - 1]:
            done = True
            self.x = [self.get_observation()]
        else:
            done = False
            self.x.append(self.get_observation())
        
        return self.x[-10:], reward, done, None  # use the last 10
