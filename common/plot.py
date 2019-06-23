import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np


class SmoothPlot():
    def __init__(self, smooth_rate=0, linewidth=1.0):
        self.smooth_rate = smooth_rate
        self.linewidth = linewidth
        self.colors = ['r', 'b', 'c', 'g', 'm', 'y', 'k', 'w']

    def plot(self, data, save_path=None, title=None, label=None):
        if type(data) == list and type(label) == list:
            for d, l, c in zip(data, label, self.colors):
                plt.plot(d, c=c, alpha=0.2, linewidth=1.0)
                plt.plot(self.smooth_moving_average(d), label=l, c=c, linewidth=self.linewidth)

        elif type(data) == np.ndarray:
            plt.plot(data, c='r', alpha=0.2, linewidth=1.0)
            plt.plot(self.smooth_moving_average(data), label=label, c='r', linewidth=self.linewidth)

        else:
            raise TypeError

        if title:
            plt.title(title)

        if label:
            plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.cla()

    def smooth_moving_average(self, arr):
        ret = [arr[0]]
        for i in range(1, len(arr)):
            ret.append(ret[i - 1] * self.smooth_rate + arr[i] * (1-self.smooth_rate))
        return ret

    def smooth_average(self, arr):
        ret = [arr[0]]
        for i in range(1, len(arr), 10):
            ret.append(float(np.mean(arr[i:i + 10])))
        return ret


def plot(folder, terms, smooth, linewidth):
    splt = SmoothPlot(smooth, linewidth)
    terms = terms.split(';')
    data = {subterm : [] for term in terms for subterm in term.split(',')}

    for root, dirs, files in os.walk(folder):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == '.log':
                for line in open(os.path.join(root, file)):
                    for x in line.split():
                        x = x.split(':')
                        if x[0] in data.keys():
                            data[x[0]].append(float(x[1]))
                break

    for term in terms:
        subterms = term.split(',')
        title = subterms[0] if len(subterms) == 1 else subterms[0].split('_')[-1]

        if len(data[subterms[0]]) > 0:
            print(f'plotting {os.path.join(folder, f"{title}.jpg")}')
            splt.plot([np.array(data[subterm]) for subterm in subterms],
                      label=[subterm for subterm in subterms],
                      save_path=os.path.join(folder, f'{title}.jpg'),
                      title=title)
