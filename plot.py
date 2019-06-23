import os

from common import args
from common.plot import plot


if __name__ == '__main__':
    for root, dirs, files in os.walk('logs'):
        for folder in dirs:
            plot(os.path.join(root, folder), args.terms, args.smooth, args.linewidth)
