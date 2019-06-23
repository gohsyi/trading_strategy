import numpy as np

from baselines.a2c.utils import discount_with_dones


class Runner(object):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, batchsize=128, gamma=0.99):
        self.env = env
        self.model = model
        self.batchsize = batchsize
        self.gamma = gamma
        
        # pre-set
        self.ob = env.reset()
        self.done = False

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        
        for n in range(self.batchsize):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            action, value = self.model.step(np.expand_dims(self.ob, 0))
            action = int(np.squeeze(action))
            value = float(np.squeeze(value))

            # Take actions in env and look the results
            ob, reward, done, _ = self.env.step(action)
            self.done = done
            self.ob = ob

            # Append the experiences
            if len(self.ob) == 10:
                mb_obs.append(self.ob)
                mb_actions.append(action)
                mb_values.append(value)
                mb_dones.append(done)
                mb_rewards.append(reward)

        # Batch of steps to batch of rollouts
        mb_obs = np.array(mb_obs, dtype=np.float32)
        mb_actions = np.array(mb_actions, dtype=np.float32)
        mb_rewards = np.array(mb_rewards, dtype=np.float32)
        mb_values = np.array(mb_values, dtype=np.float32)
        mb_dones = np.array(mb_dones, dtype=np.bool)

        if self.gamma > 0.0:
            # Discount/bootstrap
            mb_rewards = discount_with_dones(mb_rewards.tolist(), mb_dones.tolist(), self.gamma)

        return mb_obs, mb_actions, mb_rewards, mb_values
