import numpy as np

from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner


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
        self.obs = env.reset()
        self.done = False

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        
        for n in range(self.batchsize):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            action, value = self.model.step(self.obs)

            # Append the experiences
            mb_obs.append(self.obs)
            mb_actions.append(action)
            mb_values.append(value)
            mb_dones.append(self.done)

            # Take actions in env and look the results
            obs, reward, done, _ = self.env.step(actions)
            self.done = done
            self.obs = obs
            mb_dones.append(done)
            mb_rewards.append(rewards)
            
        mb_dones.append(self.done)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_dones = mb_dones[1:]

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards
        
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, None