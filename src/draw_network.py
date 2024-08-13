#!/usr/bin/env python3
import matplotlib.pyplot as plt
import sys
import csv
import tikzplotlib
import networkx as nx
from test_lib import load_graph


def draw(path, filename):

    for i in range(1, len(sys.argv)):
        plt.rcdefaults()
        sol = load_graph(path+filename)

        # plt.figure(figsize=(8, 6))
        plt.style.use('./src/paper.mplstyle')
        pos = sol.instance.coordinates
        size = [sol.instance.size[vertex] for vertex in sol.instance.vertices]
        min_size = min(size)
        diff_size = max(size)-min_size
        size = [(s-min_size)/diff_size * 0.6+1.2 for s in size]
        lines = []
        x_max = max([x for x, y in pos.values()])
        y_max = max([y for x, y in pos.values()])
        x_min = min([x for x, y in pos.values()])
        y_min = min([y for x, y in pos.values()])
        pos = {k: ((v[0]-x_min)/(x_max-x_min)*8, (v[1]-y_min) /
                   (y_max-y_min)*8) for k, v in pos.items()}
        lines.append("\\begin{tikzpicture}[scale=0.5]")
        lines.append("\\tikzstyle{city}+=[draw]")
        lines.append("\\tikzstyle{pickedge}+=[]")
        lines.append("\\tikzstyle{nonpickedge}+=[]")

        for i in range(len(sol.instance.vertices)):
            city = sol.instance.vertices[i]
            city_name = city.capitalize()
            city_name = city_name[:3].upper()
            city_name = ""
            x = pos[city][0]
            y = pos[city][1]
            sz = size[i]

            lines.append(
                f"    \\node[city, inner sep={sz}pt, label=\\scriptsize {{{city_name}}}] ({city}) at ({x},{y}) {{}};")
        for i in range(len(sol.instance.edges)):
            if sol.instance.edges[i] not in sol.pick:
                continue
            style = 'pickedge' if sol.instance.edges[i] in sol.pick else "nonpickedge"
            city1, city2 = sol.instance.edges[i]
            lines.append(
                f"    \\draw[{style}] ({city1}) -- ({city2});")
        #     lines.append(
        #         f"    \\draw[] ({sol.pick[i][0]}) -- ({sol.pick[i][1]});")
        lines.append("\\end{tikzpicture}")
        # print(lines)
        with open("{}.tex".format(path+filename[:-5]), 'w') as f:
            for line in lines:
                f.write(line + "\n")


if __name__ == '__main__':
    for i in range(1, len(sys.argv)):
        path = sys.argv[i]
        filename = path.split('/')[-1]
        path = path.split(filename)[0]
        draw(path, filename)
