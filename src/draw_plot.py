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


# def draw_broken(path, filename, object):
#     if not os.path.exists(path+filename+'.csv'):
#         print("File not found: ", path+filename+'.csv')
#         return
#     with open(path+filename+'.csv') as f:
#         plt.rcdefaults()
#         reader = csv.reader(f)
#         data = list(reader)
#         title = data[0]
#         data = data[1:]
#         data = [[float(data[i][j]) for j in range(len(data[i]))]
#                 for i in range(len(data))]
#         # plt.figure(figsize=(8, 6))
#         style = ['-', '--', ':', '-.']
#         contries = ['france', 'uk', 'germany', 'italy', 'spain']
#         country = 'unknown'
#         for i in range(len(contries)):
#             if contries[i] in path:
#                 country = contries[i]
#         if 'filter' in path:
#             country += 'filter'
#         plt.style.use('./src/paper.mplstyle')
#         r = 8
#         f, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={
#             'height_ratios': [1, r]}, sharex=True)
#         miny = 1e9
#         maxy = 0
#         if 'time' in object or 'Time' in object:
#             ax.set_yscale('log', basey=2)
#             ax2.set_yscale('log', basey=2)
#         else:
#             ax.set_yscale('linear')
#             ax2.set_yscale('linear')

#         # myinf = 1.6*max(max(data[i][j] for j in range(1, len(data[i]))
#         #                 if data[i][j] and data[i][j] < 5e7) for i in range(len(data)))
#         myinf = 1.6*max(min(data[i][j] for j in range(1, len(data[i]))
#                         if data[i][j]) for i in range(len(data)))
#         for i in range(1, len(data[0])):
#             # if i in [3, 5, 6, 8, 9]:
#             #     continue
#             y = []
#             x = []
#             for j in range(len(data)):
#                 if len(data[j]) > i:
#                     if data[j][i] == 0:
#                         continue
#                     # if data[j][i] > 5e7:
#                     if data[j][i] > 1.5*min(data[j][k] for k in range(1, len(data[j])) if data[j][k]):
#                         y.append(myinf)
#                         x.append(data[j][0])
#                         continue
#                     y.append(data[j][i])
#                     x.append(data[j][0])
#                     if data[j][i] < miny:
#                         miny = data[j][i]
#                     if data[j][i] > maxy:
#                         maxy = data[j][i]
#                 else:
#                     break

#             currmarker = markers[(i-1) % len(markers)
#                                  ] if len(data[0]) > 5 else None
#             ax.plot(x, y, label=title[i], linestyle=style[(
#                 i-1) % 4], marker=currmarker)
#             ax2.plot(x, y, label=title[i], linestyle=style[(
#                 i-1) % 4], marker=currmarker)

#             # plt.plot(x, y, label=title[i], linestyle=style[(i-1) % 4])
#         rng = maxy-miny+20
#         ax.set_ylim(myinf-0.8*rng/10, myinf+0.2*rng/10)  # outliers only
#         ax2.set_ylim(miny-0.05*rng, maxy+0.05*rng)
#         ax.spines['bottom'].set_visible(False)
#         ax2.spines['top'].set_visible(False)
#         ax.xaxis.tick_top()
#         # don't put tick labels at the top
#         ax.tick_params(labeltop=False, top=False)
#         ax.set_yticks([myinf])
#         ax.set_yticklabels(['$\infty$'])
#         ax2.xaxis.tick_bottom()

#         d = .015  # how big to make the diagonal lines in axes coordinates
#         # arguments to pass to plot, just so we don't keep repeating them
#         kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
#         ax.plot((-d, +d), (-r*d, +r*d), **kwargs)        # top-left diagonal
#         ax.plot((1 - d, 1 + d), (-r*d, +r*d), **kwargs)  # top-right diagonal

#         kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#         ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#         ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **
#                  kwargs)  # bottom-right diagonal
#         f.subplots_adjust(hspace=0.1)
#         xlabel = title[0].split('\t')[0]
#         ylabel = title[0].split('\t')[1]
#         ax2.set_xlabel(xlabel)
#         ax2.set_ylabel(ylabel)
#         ax2.legend(loc='upper left')

#         plt.savefig(path+filename + ".pdf")
#         tikzplotlib.save("{}.tex".format(path+country + '_' + filename))
#         plt.close()


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
        if 'filter' in path:
            country += 'filter'
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
        # plt.ylim(bottom=0)
        # if len(x) < 20:
        #     plt.xticks(range(int(min(x)), int(max(
        #         x))+1), [str(i)+('?'if i == 5 else '') for i in range(int(min(x)), int(max(x))+1)])

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
