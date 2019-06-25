import tensorflow as tf


def sample(logits):
    noise = tf.random_normal(tf.shape(logits))
    logits = tf.clip_by_value(logits + noise, 1e-4, 1-1e-4)
    return tf.squeeze(tf.multinomial(tf.log(logits), 1), 1)


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]
