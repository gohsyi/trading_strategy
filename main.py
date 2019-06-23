from common import args, folder
from common.util import get_logger
from common.plot import plot

from env import Env

import os
import numpy as np

from baselines.a2c.a2c import Model
from baselines.a2c.runner import Runner
from baselines.common import set_global_seeds


if __name__ == '__main__':
    env = Env()

    set_global_seeds(args.seed)

    logger = get_logger('a2c')
    logger.info(str(args))

    # Instantiate the model objects (that creates defender_model and adversary_model)
    model = Model(
        ob_size=env.ob_size,
        act_size=env.act_size,
        learning_rate=args.lr,
        latents=args.latents,
        activation=args.activation,
        optimizer=args.optimizer,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm
    )

    if args.load_path is not None:
        model.load(args.load_path)

    # Instantiate the runner object
    runner = Runner(env, model, batchsize=args.batchsize, gamma=args.gamma)

    for ep in range(args.total_epoches):
        # Get mini batch of experiences
        obs, rewards, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, rewards, actions, values)

        if (ep + 1) % args.log_interval == 0:
            avg_rewards = float(np.mean(rewards))
            avg_values = float(np.mean(values))
            idle_prob = float(np.mean(actions==1))
            long_prob = float(np.mean(actions==2))
            short_prob = float(np.mean(actions==0))

            logger.warn(
                # epoch number
                f'ep:{ep}\t'
                # losses
                f'pg_loss:{policy_loss:.3f}\tvf_loss:{value_loss:.3f}\tent_loss:{policy_entropy:.3f}\t'
                # reward and estimated rewards
                f'avg_rew:{avg_rewards:.2f}\tavg_val:{avg_values:.2f}\t'
                # action proportion
                f'long_prob:{long_prob:.2f}\tshort_prob:{short_prob:.2f}\tidle_prob:{idle_prob:.2f}'
            )

        if (ep + 1) % args.save_interval == 0:
            model.save(os.path.join(folder, f'model_{ep}.ckpt'))

    plot(folder, args.terms, args.smooth, args.linewidth)
