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

    def __init__(self, env, d_model, a_model, nsteps=5, gamma=0.99):
        self.env = env
        self.d_model = d_model
        self.a_model = a_model
        self.obs = env.reset()
        self.nsteps = nsteps
        self.dones = [False]

        self.gamma = gamma

    def run(self):
        """
        Make a mini batch of experiences
        :return:
            mb_obs: (batch_size x ob_size), observations of both defender and attacker
            (mb_d_rewards, mb_a_rewards): (batch_size x 1, batch_size x 1), rewards of attacker
            (mb_d_actions, mb_a_actions): (batch_size x 1, batch_size x 1), actions of attacker
            (mb_d_values, mb_a_values): (batch_size x 1, batch_size x 1), estimated value of attacker
            epinfos: other infos (useless for now)
        """

        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_d_rewards, mb_dones = [],[],[]
        mb_d_actions, mb_a_actions, mb_d_values, mb_a_values = [],[],[],[]
        epinfos = []
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            d_actions, d_values = self.d_model.step(self.obs)
            a_actions, a_values = self.a_model.step(self.obs)
            d_actions = np.squeeze(d_actions)
            a_actions = np.squeeze(a_actions)
            d_values = np.squeeze(d_values)
            a_values = np.squeeze(a_values)

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_d_actions.append(d_actions)
            mb_a_actions.append(a_actions)
            mb_d_values.append(d_values)
            mb_a_values.append(a_values)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            actions = (d_actions, a_actions)
            obs, rewards, dones, infos = self.env.step(actions)
            self.dones = dones
            self.obs = obs
            mb_d_rewards.append(rewards)

        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32)
        mb_d_rewards = np.asarray(mb_d_rewards, dtype=np.float32)
        mb_a_rewards = -mb_d_rewards
        mb_d_actions = np.asarray(mb_d_actions, dtype=np.int32)
        mb_a_actions = np.asarray(mb_a_actions, dtype=np.int32)
        mb_d_values = np.asarray(mb_d_values, dtype=np.float32)
        mb_a_values = np.asarray(mb_a_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # TODO add bootstrap
        # if self.gamma > 0.0:
        #     # Discount/bootstrap off value fn for defender
        #     last_values = self.d_model.value(self.obs).tolist()
        #     for n, (rewards, dones, value) in enumerate(zip(mb_d_rewards, mb_dones, last_values)):
        #         rewards = rewards.tolist()
        #         dones = dones.tolist()
        #         if dones[-1] == 0:
        #             rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
        #         else:
        #             rewards = discount_with_dones(rewards, dones, self.gamma)
        #
        #         mb_d_rewards[n] = rewards
        #
        #     # Discount/bootstrap off value fn for attacker
        #     last_values = self.a_model.value(self.obs).tolist()
        #     for n, (rewards, dones, value) in enumerate(zip(mb_a_rewards, mb_dones, last_values)):
        #         rewards = rewards.tolist()
        #         dones = dones.tolist()
        #         if dones[-1] == 0:
        #             rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
        #         else:
        #             rewards = discount_with_dones(rewards, dones, self.gamma)
        #
        #         mb_a_rewards[n] = rewards

        return mb_obs, \
               (mb_d_rewards, mb_a_rewards), \
               (mb_d_actions, mb_a_actions), \
               (mb_d_values, mb_a_values), \
               epinfos
