import tensorflow as tf

from baselines.common import layers
from baselines.a2c.utils import sample


class Policy(object):
    """
    Policy network for A2C
    """

    def __init__(
            self,
            observations,  # observations, placeholder
            act_size,  # size of action space
            latents,  # hidden layer dims of policy network
            vf_latents,  # hidden layer dims of value network
            activation):  # activation function,

        with tf.variable_scope('actor'):
            latent = layers.lstm(observations, latents, activation)
            logits = layers.dense(latent, act_size)  # to compute loss
            action = sample(tf.nn.softmax(logits))

        with tf.variable_scope('critic'):
            vf_latent = layers.lstm(observations, vf_latents, activation)
            vf = tf.squeeze(layers.dense(vf_latent, 1), -1)

        def neglogp(actions):
            return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.clip_by_value(logits, tf.constant(-1e4), tf.constant(1e4)),
                labels=actions,
            )

        def entropy():
            a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, 1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


        self.vf = vf
        self.action = action
        self.neglogp = neglogp
        self.entropy = entropy

        # for debug
        self.latent = latent
        self.vf_latent = vf_latent
        self.logits = logits


def build_policy(observations, act_size, latents, vf_latents, activation):
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
                  latents=latents,
                  vf_latents=vf_latents,
                  activation=activation)
