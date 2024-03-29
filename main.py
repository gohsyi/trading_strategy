from common import args, folder
from common.util import get_logger
from common.plot import plot

from env import Env

import os
import numpy as np

from rl.a2c.a2c import Model
from rl.a2c.runner import Runner
from rl.common import set_global_seeds

from baselines.stochastic import Model as Stochastic
from baselines.rule import Model as Rule

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    set_global_seeds(args.seed)

    logger = get_logger('trading')
    logger.info(str(args))

    env = Env('train')

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

    """ train """
    for ep in range(args.total_epoches):
        # Get mini batch of experiences
        obs, actions, rewards, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, actions, rewards, values)

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
                f'avg_rew:{avg_rewards:.3f}\tavg_val:{avg_values:.3f}\t'
                # action proportion
                f'long_prob:{long_prob:.2f}\tshort_prob:{short_prob:.2f}\tidle_prob:{idle_prob:.2f}\t'
            )

        if (ep + 1) % args.save_interval == 0:
            model.save(os.path.join(folder, f'model_{ep}.ckpt'))


    models = [
        model,
        Stochastic(env.act_size),
        Rule(),
    ]

    model_names = [
        'a2c',
        'stochastic',
        'supervised',
    ]

    """ test """
    for model, model_name in zip(models, model_names):
        profits = [0]
        avg_buys, avg_sells = [], []
        assets = []
        env = Env('val')
        observation = env.reset()
        done = False
        step = 0

        while not done:
            action, _ = model.step([observation])
            observation, reward, done, _ = env.step(int(action))
            # avg_buy, avg_sell = env.get_avg_prices()
            profits.append(profits[-1] + reward)
            # avg_buys.append(avg_buy)
            # avg_sells.append(avg_sell)
            assets.append(env.get_asset())
            # logger.warn(f'step:{step}\tval_act:{int(action)}\tval_rew:{reward}')
            step += 1

        avg_buys, buy_prob = env.get_avg_buying()
        avg_sells, sell_prob = env.get_avg_selling()

        logger.warn(f'model:{model_name}\'\tbuy/sell\t'
                    f'price:{avg_buys}/{avg_sells}\t'
                    f'prob:{buy_prob}/{sell_prob}')

        plt.plot(assets, label=f'assets of {model_name}', linewidth=args.linewidth)

    plt.legend()
    plt.savefig(os.path.join(folder, 'val.jpg'))
    plt.cla()


    plot(folder, args.terms, args.smooth, args.linewidth)
