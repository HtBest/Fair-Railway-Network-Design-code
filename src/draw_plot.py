#!/usr/bin/env python3
import copy
import os
import matplotlib.pyplot as plt
import sys
import csv
import tikzplotlib
from math import *
# from brokenaxes import brokenaxes


markers = ['+', 'x', '*']


def draw(path, filename, smoothing=3):
    if 'time' in filename or 'Time' in filename:
        plt.yscale('log', basey=2)
    else:
        plt.yscale('linear')
    if not os.path.exists(path+filename):
        print("File not found: ", path+filename)
        return
    with open(path+filename) as f:
        plt.rcdefaults()
        reader = csv.reader(f)
        data = list(reader)
        title = data[0]
        data = data[1:]
        # plt.figure(figsize=(8, 6))
        style = ['-', '--', ':', '-.']
        contries = ['france', 'uk', 'germany', 'italy',
                    'spain', 'us', 'canada', 'russia', 'brazil']
        country = 'unknown'
        for i in range(len(contries)):
            if contries[i] in path:
                country = contries[i]
        if 'compare' in path:
            country += 'compare'
        plt.style.use('./src/paper.mplstyle')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(1, len(data[0])):
            y = []
            x = []
            drawlabel = True
            for j in range(len(data)):
                if len(data[j]) > i:
                    if float(data[j][i]) == 0:
                        continue
                    if float(data[j][i]) > 5e7:
                        plt.clf()
                        assert (False)
                    y.append(float(data[j][i]))
                    x.append(float(data[j][0]))
                else:
                    break
            lsy = copy.deepcopy(y)
            for j in range(len(y)):
                newr = min(smoothing, j, len(y)-j-1)
                y[j] = sum(lsy[j-newr:j+newr+1])/(2*newr+1)
            if len(x):
                currmarker = markers[(i-1) % len(markers)
                                     ] if len(data[0]) > 50 else None
                plt.plot(x, y, label=title[i] if drawlabel else None, linestyle=style[(
                    i-1) % 4], marker=currmarker, color=colors[(i-1) % len(colors)], linewidth=0.5)

        xlabel = title[0].split('\t')[0]
        ylabel = title[0].split('\t')[1]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(path+filename[:-4] + ".pdf")
        tikzplotlib.save("{}.tex".format(path+country + '_' + filename[:-4]))
        plt.close()


if __name__ == '__main__':

    for i in range(1, len(sys.argv)):
        filename = sys.argv[i]
        files = [f for f in os.listdir(filename) if f.endswith('.csv')]
        for file in files:
            draw(filename + '/', file, 3)
