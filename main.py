from common.argparser import args
from env import Env
from baselines.a2c.a2c import learn


if __name__ == '__main__':
    env = Env(args.train_path)
    model = learn(
        env=env,
        seed=1953,
        nsteps=args.batchsize,
        total_epoches=int(8e6),
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=1e-4,
        gamma=0.99,
        log_interval=1,
        load_path=None,
    )
