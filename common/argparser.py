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
parser.add_argument('-batchsize', type=int, default=64)
parser.add_argument('-latents', type=str, default='256')
parser.add_argument('-total_epoches', type=int, default=int(1e5))
parser.add_argument('-vf_coef', type=float, default=0.1)
parser.add_argument('-ent_coef', type=float, default=0.01)
parser.add_argument('-max_grad_norm', type=float, default=0.5)
parser.add_argument('-activation', type=str, default='relu',
                    help='relu/sigmoid/elu/tanh')
parser.add_argument('-optimizer', type=str, default='adam',
                    help='adam/adagrad/gd/rms/momentum')

# path setting
parser.add_argument('-train_path', type=str, default='../DataSet/TrainSet.csv')
parser.add_argument('-test_path', type=str, default='../DataSet/TestSet.csv')

args = parser.parse_args()

abstract = '{}_{}_{}_lr{}{}hid{}_bs{}_ep{}_grad{}_vf{}_ent{}_seed{}'.format(
    args.model,
    args.activation,
    args.optimizer,
    args.lr,
    '_decay_' if args.lr_decay else '_',
    args.latents,
    args.batchsize,
    args.total_epoches,
    args.max_grad_norm,
    args.vf_coef,
    args.ent_coef,
    args.seed,
)

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
