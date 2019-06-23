import os
import numpy as np

from common import args
from common.plot import SmoothPlot, plot


if __name__ == '__main__':
    splt = SmoothPlot(args.smooth, args.linewidth)
    terms = args.terms.split(';')

    for root, dirs, files in os.walk('logs'):
        for folder in dirs:
            plot(os.path.join(root, folder), args.terms, args.smooth, args.linewidth)

        for file in files:
            _, ext = os.path.splitext(file)
            if ext == '.log':
                data = []
                for line in open(os.path.join(root, file)):
                    for x in line.split():
                        x = x.split(':')
                        if x[0] == 'val_rew':
                            data.append(float(x[1]))

                for i in range(1, len(data)):
                    data[i] += data[i - 1]  # accumulated profits

                if len(data) > 0:
                    path = os.path.join(root, 'profits.jpg')
                    print(f'plotting {path}')
                    splt.plot(np.array(data), title='profits',
                              save_path=os.path.join(root, 'profits.jpg'))
