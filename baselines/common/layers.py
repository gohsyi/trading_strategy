import tensorflow as tf


def dense(inputs, units, name=None, activation=None):
    """
    construct a fully-connected layer
    :param inputs: inputs of fc
    :param units: number of outputs
    :param activation: activation function
    :return: corresponding created FC layer
    """

    return tf.layers.dense(
        inputs, units,
        # name=name,
        activation=activation,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.5)
    )


def mlp(x, latents, activation=None):
    """
    construct a multi-layer perception
    :param x: input
    :param latents: latent sizes
    :param activation: activation function
    :return: corresponding created MLP
    """

    last_latent = x
    for i, hdim in enumerate(latents):
        last_latent = dense(last_latent, hdim, 'layer%i' % i, activation)
    return last_latent
