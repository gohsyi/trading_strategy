import functools
import numpy as np
import tensorflow as tf

from tensorflow import losses


from common.argparser import args, abstract
from common.util import get_logger

from baselines.common import tf_util
from baselines.common import set_global_seeds

from baselines.a2c.policy import build_policy
from baselines.a2c.utils import find_trainable_variables
from baselines.a2c.runner import Runner


class Model(object):
    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """

    def __init__(
            self,
            name,  # name of this model
            env,  # environment
            latents,  # network hidden layer sizes
            lr=1e-5,  # learning rate
            activation='relu',  # activation function
            optimizer='adam',  # optimization function
            vf_coef=0.1,  # vf_loss weight
            ent_coef=0.01,  # ent_loss weight
            max_grad_norm=0.5,  # grad normalization
            total_epoches=int(80e6),  # total number of epoches
            log_interval=100):  # how frequently the logs are printed out

        sess = tf_util.get_session()

        # output to both file and console
        logger = get_logger(name)
        output = logger.info

        activation = tf_util.get_activation(activation)
        optimizer = tf_util.get_optimizer(optimizer)

        lr = tf.train.polynomial_decay(
            learning_rate=lr,
            global_step=tf.train.get_or_create_global_step(),
            decay_steps=total_epoches,
            end_learning_rate=lr/10,
        )

        ob_size = env.ob_size
        act_size = env.act_size
        n_actions = env.n_actions

        # placeholders for use
        X = tf.placeholder(tf.float32, [None, ob_size], 'observation')
        A = tf.placeholder(tf.int32, [None, act_size], 'action')
        ADV = tf.placeholder(tf.float32, [None], 'advantage')
        R = tf.placeholder(tf.float32, [None], 'reward')

        with tf.variable_scope(name):
            policy = build_policy(
                observations=X,
                act_size=act_size,
                n_actions=n_actions,
                latents=latents,
                vf_latents=latents,
                activation=activation
            )

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = policy.neglogp
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(policy.entropy)

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(policy.vf), R)

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # gradients and optimizer
        params = find_trainable_variables(name)
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        # 3. Make op for one policy and value update step of A2C
        trainer = optimizer(learning_rate=lr)

        _train = trainer.apply_gradients(grads)

        def step(obs):
            return sess.run([policy.action, policy.vf], feed_dict={
                X: np.reshape(obs, (-1, ob_size))
            })

        def value(obs):
            return sess.run(policy.vf, feed_dict={
                X: np.reshape(obs, (-1, ob_size))
            })

        def train(ep, obs, rewards, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values

            td_map = {X:obs, A:actions, ADV:advs, R:rewards}
            policy_loss, value_loss, policy_entropy, cur_lr, _ = sess.run(
                [pg_loss, vf_loss, entropy, lr, _train],
                td_map
            )

            if (ep + 1) % log_interval == 0:
                avg_rewards = float(np.mean(rewards))
                avg_values = float(np.mean(values))
                avg_advs = float(np.mean(advs))

                output(f'ep:%i\tlr:%f\t' % (ep, cur_lr) +
                       f'pg_loss:%.3f\tvf_loss:%.3f\tent_loss:%.3f\t' % (policy_loss, value_loss, policy_entropy) +
                       f'avg_rew:%.2f\tavg_val:%.2f\tavg_adv:%.2f\t' % (avg_rewards, avg_values, avg_advs))

            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.step = step
        self.value = value
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
    env,
    seed=None,
    nsteps=5,
    total_epoches=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    gamma=0.99,
    log_interval=1,
    load_paths=None):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)

    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_epoches:      int, total number of epoches to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate (default: 7e-4)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    '''


    set_global_seeds(seed)

    # Instantiate the model objects (that creates defender_model and adversary_model)
    d_model = Model(
        name='defender',
        env=env,
        lr=lr,
        latents=args.latents,
        activation=args.activation,
        optimizer=args.optimizer,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        total_epoches=total_epoches,
        log_interval=log_interval)

    a_model = Model(
        name='attacker',
        env=env,
        lr=lr,
        latents=args.latents,
        activation=args.activation,
        optimizer=args.optimizer,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        total_epoches=total_epoches,
        log_interval=log_interval)

    if load_paths is not None:
        d_model.load(load_paths[0])
        a_model.load(load_paths[1])

    # Instantiate the runner object
    runner = Runner(env, d_model, a_model, nsteps=nsteps, gamma=gamma)

    for ep in range(total_epoches):
        # Get mini batch of experiences
        obs, rewards, actions, values, epinfos = runner.run()
        d_rewards, a_rewards = rewards
        d_actions, a_actions = actions
        d_values, a_values = values

        d_model.train(ep, obs, d_rewards, d_actions, d_values)
        a_model.train(ep, obs, a_rewards, a_actions, a_values)

    return d_model, a_model
