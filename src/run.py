#!/usr/bin/env python3
import copy
from typing import List
import matplotlib.pyplot as plt
import random


import graph as g
import sys

from test_items import solve_all, compare_algorithm
from utils import delete_edges_iterative


def auto_test():
    maps = ['us', 'canada', 'russia', 'brazil',
            'uk', 'italy', 'germany', 'spain', 'france']
    if len(sys.argv) > 1:
        maps = [sys.argv[1]]

    for m in maps:
        print("Running test on map:", m)
        input_file = 'map_instances/'+m+'.city'
        G = g.Graph()
        G.load(input_file)
        m = len(G.graph.edges)
        num_delete = m//2
        G, _ = delete_edges_iterative(G, num_delete, [1, 2, 4, 6, 10])
        print("Number of edges left:", len(G.graph.edges))
        print("Test on all budgets:")
        solve_all(G, filename=input_file,
                  P=[3], algo='local_search')

    for m in maps:
        print("Running test on map: ", m)
        input_file = 'map_instances/'+m+'.city'
        print("Test on filter:")
        arr = [5, 6, 7, 8, 9, 10, 11, 12, 13]
        compare_algorithm(arr, 1,  [
            'local_search', 'exact'], filename=input_file)


if __name__ == '__main__':
    auto_test()
