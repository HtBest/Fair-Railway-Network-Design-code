import copy
import datetime
import json
import os
from typing import List
import matplotlib.pyplot as plt
import csv
from draw_network import draw as draw_network
from draw_plot import draw as draw_plot
from statistic import generate_csv

import numpy as np
import network_design as nd
import graph as g
import networkx as nx

from test_lib import Wrapper
from utils import delete_edges_iterative, get_budgets, pname


def solve_all(G: g.Graph, filename, P: List, algo: str):
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    data = G.to_dict()
    pos = {k: (v['x'], v['y']) for (k, v) in data['vertices'].items()}
    sizes = [v['size'] for v in data['vertices'].values()]
    min_size = min(sizes)
    diff_size = max(sizes)-min_size
    new_size = [(s-min_size)/diff_size * 800+200 for s in sizes]
    Gs = {p: G for p in P}
    budgets = get_budgets(G)
    init_sols = {}
    cities = list(G.graph.nodes)
    for p in P:
        init_sols[p] = []
        _, edges = delete_edges_iterative(Gs[p], len(Gs[p].graph.edges), [p])
        x = np.zeros((len(G.graph.nodes), len(G.graph.nodes)))
        cost = 0
        for i in range(len(edges)):
            u, v = edges[i]
            x[u, v] = x[v, u] = 1
            cost += Gs[p].graph[cities[u]][cities[v]]['distance']
            init_sols[p].append((copy.deepcopy(x), cost))

    def get_init(budget: int, p: int):
        ans = None
        for i in range(len(init_sols[p])):
            if init_sols[p][i][1] <= budget:
                ans = init_sols[p][i][0]
            else:
                break
        return ans
    instance_groups = [
        [nd.Instance(Gs[p].to_dict(), budget=budget, egal_bias=p, algo=algo, P=P, init_solution=get_init(budget, p)) for p in P] for budget in budgets]
    print('range:', budgets[0], budgets[-1])
    wrapper = Wrapper()
    data = wrapper.eval(instance_groups)

    graphfilename = filename.split('/')[-1].split('.')[0]+'_budget'
    if not os.path.exists('results/'+current_time+graphfilename):
        os.makedirs('results/'+current_time+graphfilename)

    for i in range(len(data)):
        for j in range(len(data[i])):
            budget = budgets[i]
            plt.clf()
            plt.figure(figsize=wrapper.figsize)
            nx.draw_networkx_nodes(G.graph, pos, node_size=new_size,
                                   node_color='lightblue')
            nx.draw_networkx_edges(G.graph, pos, edgelist=G.graph.edges(),
                                   width=0.5, edge_color=(0, 1, 0, 0.3), style='dashed')
            nx.draw_networkx_edges(G.graph, pos, edgelist=data[i][j].pick,
                                   width=2, edge_color='green')
            nx.draw_networkx_labels(G.graph, pos, font_size=15,
                                    font_family="sans-serif")
            plt.axis("off")
            plt.figtext(0.5, 0.9, "p="+pname(P[j]), ha="center", fontsize=20, bbox={
                        "facecolor": "orange", "alpha": 0.5, "pad": 5})

            filename = str(budget)+' p='+pname(P[j])
            plt.savefig(
                'results/'+current_time+graphfilename+'/'+filename+'.png')
            plt.close()
            with open('results/'+current_time+graphfilename+'/'+filename+'.json', 'w') as f:
                json.dump(data[i][j].to_dict(), f)
            draw_network('results/'+current_time+graphfilename +
                         '/', filename+'.json')
    generate_csv('results/'+current_time+graphfilename)


def compare_algorithm(ns, p, algos, filename):
    G = g.Graph()
    G.load(filename)
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    data = G.to_dict()
    instance_groups = []
    for algo in algos:
        instance_groups.append([])
        for n in ns:
            instance_groups[-1].append([])
            G = g.Graph()
            G.load(filename, n)
            budgets = get_budgets(G, 60)
            for budget in budgets:
                instance_groups[-1][-1].append(nd.Instance(
                    G.to_dict(), budget=budget, egal_bias=p, P=[p], algo=algo))

    wrapper = Wrapper()
    data = wrapper.eval(instance_groups)
    graphfilename = filename.split('/')[-1].split('.')[0]+'_filter'
    if not os.path.exists('results/'+current_time+graphfilename):
        os.makedirs('results/'+current_time+graphfilename)
    # plot and csv
    keys = ['time', 'norms', 'optimal_rate']
    keynames = ['Time', 'Relative $p=' +
                pname(p)+'$-social cost', 'Optimal Rate']

    def algoname(
        algo): return algo == 'local_search' and 'Local Search' or 'Exact Algorithm'
    for k, name in zip(keys, keynames):
        if k == 'norms':
            def get_value(d): return getattr(d, k)[p]
        elif k == 'optimal_rate':
            def get_value(d): return getattr(d, 'norms')[p]
        else:
            def get_value(d): return getattr(d, k)
        csv_data = []
        row = ['Number of Vertices\t'+name]+[algoname(algo) for algo in algos]
        csv_data.append(row)
        for n in ns:
            csv_data.append([n])
        if k == 'optimal_rate':
            y = []
            x = []
            csv_data[0] = csv_data[0][:2]
            for n in range(len(ns)):
                optimal = 0
                for ith, budget in enumerate(budgets):
                    optimal += get_value(data[0][n][ith]
                                         ) == get_value(data[1][n][ith])
                y.append(optimal/len(budgets))
                x.append(ns[n])

            for xx, yy in zip(x, y):
                ix = 0
                for i in range(len(csv_data)):
                    if csv_data[i][0] == xx:
                        ix = i
                        break
                csv_data[ix].append(yy)
        else:
            for i in range(len(algos)):
                y = []
                x = []
                for n in range(len(ns)):
                    sum_cost = 0
                    for ith, budget in enumerate(budgets):
                        sum_cost += get_value(data[i][n][ith])
                    y.append(sum_cost/len(budgets))
                    x.append(ns[n])
                for xx, yy in zip(x, y):
                    ix = 0
                    for i in range(len(csv_data)):
                        if csv_data[i][0] == xx:
                            ix = i
                            break
                    csv_data[ix].append(yy)
        if k == 'norms':
            for i in range(1, len(csv_data)):
                csv_data[i][1] = csv_data[i][1]/csv_data[i][2]
            for i in range(len(csv_data)):
                csv_data[i] = csv_data[i][:2]
        with open('results/'+current_time+graphfilename+'/'+k+f' p={p}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        draw_plot('results/'+current_time +
                  graphfilename+'/', k+f' p={p}.csv', 0)


def solve_one(G, budget, p=1, algo='local_search', save=False, params={}):
    P = [p]
    data = G.to_dict()
    pos = {k: (v['x'], v['y']) for (k, v) in data['vertices'].items()}
    sizes = [v['size'] for v in data['vertices'].values()]
    min_size = min(sizes)
    diff_size = max(sizes)-min_size
    new_size = [(s-min_size)/diff_size * 800+200 for s in sizes]
    group = [nd.Instance(data, budget=budget, egal_bias=p, P=P,
                         algo=algo) for p in P]
    wrapper = Wrapper()
    data = wrapper.eval(group, params=params)
    if (save):
        for i in range(len(data)):
            plt.figure(figsize=wrapper.figsize)
            nx.draw_networkx_nodes(G.graph, pos, node_size=new_size,
                                   node_color='lightblue')
            nx.draw_networkx_edges(G.graph, pos, edgelist=G.graph.edges(),
                                   width=0.5, edge_color=(0, 1, 0, 0.2), style='dashed')
            nx.draw_networkx_edges(G.graph, pos, edgelist=data[i].pick,
                                   width=2, edge_color='green')
            nx.draw_networkx_labels(G.graph, pos, font_size=15,
                                    font_family="sans-serif")
            plt.axis("off")
            plt.figtext(0.5, 0.9, "p="+pname(P[i]), ha="center", fontsize=20, bbox={
                        "facecolor": "orange", "alpha": 0.5, "pad": 5})

            filename = str(budget)+' p='+pname(P[i])
            plt.savefig(filename+'.png')
            plt.close()

            with open(filename+'.json', 'w') as f:
                json.dump(data[i].to_dict(), f)
            draw_network('./', filename+'.json')
    return data[0].pick, data[0].norms


def draw_edge_deletion(G, edges):

    data = G.to_dict()
    pos = {k: (v['x'], v['y']) for (k, v) in data['vertices'].items()}
    sizes = [v['size'] for v in data['vertices'].values()]
    min_size = min(sizes)
    diff_size = max(sizes)-min_size
    new_size = [(s-min_size)/diff_size * 800+200 for s in sizes]

    plt.clf()
    plt.figure(figsize=(18, 16))
    nx.draw_networkx_nodes(G.graph, pos, node_size=new_size,
                           node_color='lightblue')
    nx.draw_networkx_edges(G.graph, pos, edgelist=G.graph.edges(),
                           width=2, edge_color='green')
    nx.draw_networkx_edges(G.graph, pos, edgelist=edges,
                           width=2, edge_color='red', style='dashed')
    nx.draw_networkx_edge_labels(G.graph, pos, edge_labels={(
        u, v): edges.index((u, v))+1 for u, v in edges}, font_size=15)
    nx.draw_networkx_labels(G.graph, pos, font_size=15,
                            font_family="sans-serif")
    plt.axis("off")
    plt.savefig('delete_edges_iterative'+str(len(edges))+'.png')
