import tensorflow as tf

from baselines.common import layers, sample_k


class Policy(object):
    """
    Policy network for A2C
    """

    def __init__(
            self,
            observations,  # observations, placeholder
            act_size,  # size of action space
            n_actions,  # number of actions we should take in each step
            latents,  # hidden layer dims of policy network
            vf_latents,  # hidden layer dims of value network
            activation):  # activation function,

        latent = layers.mlp(observations, latents, activation)
        logits = layers.dense(latent, act_size)  # to compute loss
        pi = tf.nn.softmax(logits)

        action = sample_k(logits, n_actions)

        vf_latent = layers.mlp(observations, vf_latents, activation)
        vf = tf.squeeze(layers.dense(vf_latent, 1), -1)

        def neglogp():
            return tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=tf.clip_by_value(logits, tf.constant(-1e4), tf.constant(1e4)),
                labels=tf.stop_gradient(action)
            )

        def entropy():
            a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, 1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


        self.pi = pi
        self.vf = vf
        self.action = action
        self.neglogp = neglogp()
        self.entropy = entropy()


def build_policy(observations, act_size, n_actions, latents, vf_latents, activation=None):
    """
    build a policy with given params
    :param observations:
    :param act_size:
    :param latents:
    :param vf_latents:
    :param activation:
    :return: Policy, a class object
    """

    return Policy(observations=observations,
                  act_size=act_size,
                  n_actions=n_actions,
                  latents=latents,
                  vf_latents=vf_latents,
                  activation=activation)
