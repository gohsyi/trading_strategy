import os
import argparse


parser = argparse.ArgumentParser()

# for jupyter-notebook
parser.add_argument('-f', type=str)

# gpu device
parser.add_argument('-gpu', type=str, default='-1')

# global random seed
parser.add_argument('-seed', type=int, default=0)

# algorithm setting
parser.add_argument('-model', type=str, default='a2c')

# experiment setting
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-lr_decay', action='store_true', default=False)
parser.add_argument('-gamma', type=float, default=0.9, help='reward decay factor')
parser.add_argument('-batchsize', type=int, default=64)
parser.add_argument('-latents', type=str, default='4,4')
parser.add_argument('-total_epoches', type=int, default=int(1e5))
parser.add_argument('-vf_coef', type=float, default=0.1)
parser.add_argument('-ent_coef', type=float, default=0.01)
parser.add_argument('-max_grad_norm', type=float, default=0.5)
parser.add_argument('-activation', type=str, default='tanh',
                    help='relu/sigmoid/elu/tanh')
parser.add_argument('-optimizer', type=str, default='adam',
                    help='adam/adagrad/gd/rms/momentum')

# path setting
parser.add_argument('-note', type=str, default='test')
parser.add_argument('-train_path', type=str, default='../DataSet/TrainSet.csv')
parser.add_argument('-test_path', type=str, default='../DataSet/ValSet.csv')
parser.add_argument('-load_path', type=str, default=None)
parser.add_argument('-pred_path', type=str, default='../XGBoosting_GBlinear.model')

# environment setting
parser.add_argument('-max_position', type=int, default=5)

# save results
parser.add_argument('-log_interval', type=int, default=1)
parser.add_argument('-save_interval', type=int, default=None)
parser.add_argument('-linewidth', type=float, default=0.75)
parser.add_argument('-smooth', type=float, default=0, help='moving average smooth rate')
parser.add_argument('-terms', type=str, default='pg_loss;vf_loss;ent_loss;avg_rew,avg_val;'
                                                'long_prob,short_prob,idle_prob;val_rew')

args = parser.parse_args()
args.save_interval = args.save_interval or (args.total_epoches // 10)
args.latents = list(map(int, args.latents.split(',')))

folder = os.path.join('logs', args.note)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
