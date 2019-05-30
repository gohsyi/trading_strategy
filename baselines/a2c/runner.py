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
            action, value = self.model.step(self.ob)

            # Append the experiences
            mb_obs.append(self.ob)
            mb_actions.append(action)
            mb_values.append(value)

            # Take actions in env and look the results
            ob, reward, done, _ = self.env.step(action)
            self.done = done
            self.ob = ob
            mb_dones.append(done)
            mb_rewards.append(reward)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        if self.gamma > 0.0:
            # Discount/bootstrap
            mb_rewards = discount_with_dones(mb_rewards.tolist(), mb_dones.tolist(), self.gamma)

        return mb_obs, mb_rewards, mb_actions, mb_values
