from common.argparser import args
from env import Env
from baselines.a2c.a2c import learn


if __name__ == '__main__':
    env = Env(args.train_path)
    model = learn(
        env=env,
        seed=1953,
        batch_size=args.batchsize,
        total_epoches=args.total_epoches,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        gamma=0.99,
        log_interval=1,
        load_path=None,
    )
