import copy
import math
from typing import List

from matplotlib import pyplot as plt
import graph as g
import network_design as nd
from solver import Solver


def pname(p):
    return str(p) if p > 0 else 'inf'


def delete_edges_iterative(G: g.Graph, num: int, egal_biases: List[int]):
    '''
    The input graph G is not modified.
    The output graph is a new graph with some edges deleted.
    '''
    s = Solver(nd.Instance(G.to_dict(), egal_bias=egal_biases[0]))

    res = s.delete_edges_iterative(num, egal_biases)
    resG = copy.deepcopy(G)
    resG.apply_matrix(s.x)
    return resG, res


def add_edges_iterative(G: g.Graph, num: int, egal_biases: List[int]):
    s = Solver(nd.Instance(G.to_dict(), egal_bias=egal_biases[0]))

    res = s.add_edges_iterative(num, egal_biases)
    return None, res  # Plz do not use the first parameter for now, it is not implemented


def get_budgets(G: g.Graph, num_pieces: int = 301):
    num_pieces1 = num_pieces//6
    num_pieces2 = num_pieces-num_pieces1-1
    sum_distance = sum([d['distance']
                       for u, v, d in G.graph.edges(data=True)])
    mst = G.mst()
    shortest_edge = min([d['distance']
                        for u, v, d in G.graph.edges(data=True)])
    base = 10 ** (math.log10(sum_distance/mst) / num_pieces2)
    budgets = []
    if num_pieces1 > 0:
        gap = max(1, (mst-shortest_edge)/num_pieces1)
        budgets = [round(shortest_edge+gap*i) for i in range(num_pieces1)]
    budgets += [round(mst*(base**i))
                for i in range(num_pieces2)]+[int(sum_distance+0.99)]
    return budgets


def local_search(G, budget, p=1, params={}):
    P = [p]
    data = G.to_dict()
    instance = nd.Instance(data, budget=budget,
                           egal_bias=p, P=P, algo='local_search')
    solver = Solver(instance)
    res1, tm = solver.run(**params)
    if res1 is None:
        return None
    sol = nd.Solution(instance, matrix=res1, time_used=tm)
    return sol.pick, sol.norms
