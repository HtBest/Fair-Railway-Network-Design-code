import copy
import math
import random
import sys
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import network_design as nd
import networkx as nx


class Solver:
    def __init__(self, instance: nd.Instance):
        # number_of_cities, populations, distances,
        self.number_of_cities = len(instance.vertices)
        id = {}
        self.algo = instance.algo
        for i in instance.vertices:
            id[i] = len(id)
        self.populations = []
        self.distances = [[-1 for _ in range(self.number_of_cities)] for _ in range(
            self.number_of_cities)]
        self.egal_bias = instance.egal_bias
        self.budget = instance.budget

        self.x = np.zeros((self.number_of_cities, self.number_of_cities))
        self.init_solution = instance.init_solution

        for x, y in instance.edges:
            a = id[x]
            b = id[y]
            self.distances[a][b] = instance.distance[(x, y)]
            self.distances[b][a] = instance.distance[(x, y)]
            self.x[a][b] = -1
            self.x[b][a] = -1

        for i in range(self.number_of_cities):
            self.distances[i][i] = 0
            self.x[i][i] = 0
        for i in range(self.number_of_cities):
            self.populations.append([])
            for j in range(self.number_of_cities):
                if i == j:
                    trip = 0
                elif (
                        instance.vertices[i], instance.vertices[j]) in instance.trips:
                    trip = instance.trips[(
                        instance.vertices[i], instance.vertices[j])]
                elif (
                        instance.vertices[j], instance.vertices[i]) in instance.trips:
                    trip = instance.trips[(
                        instance.vertices[j], instance.vertices[i])]
                else:
                    trip = 0
                self.populations[i].append(trip)
        self.create_commuters()
        self.cntFolyd = 0
        self.of = self.object_function(self.egal_bias)

    # a function usefull for initializing a matrix of commuters from city to city

    def create_commuters(self,):
        self.commuter_matrix = [[0 for _ in range(self.number_of_cities)]
                                for _ in range(self.number_of_cities)]
        for i in range(self.number_of_cities):
            for j in range(self.number_of_cities):
                if (i != j):
                    self.commuter_matrix[i][j] = int(self.populations[i][j])

    # determines if the graph is connected
    def is_connected(self, x):
        state = np.array([0 for _ in range(self.number_of_cities)])
        state = self.explore(x, state, 0)
        if (0 in state):
            return False
        return True

    def explore(self, x, state, node):
        state[node] = 1
        for s in range(self.number_of_cities):
            if ((x[node][s] == 1) and (state[s] == 0)):
                state = self.explore(x, state, s)
        return state

    def evaluate_budget(self, x):
        return sum(sum(abs(x) * self.distances))/2

    def compute_distance(self, graph) -> List[List[int]]:
        self.cntFolyd += 1
        # Number of vertices in the graph
        V = len(graph)
        # Initialize dist and path matrices with initial values
        dist = [[float('inf') for _ in range(V)] for _ in range(V)]
        g = nx.Graph()

        for i in range(V):
            g.add_node(i)
            for j in range(V):
                if graph[i][j] != 0:
                    g.add_edge(i, j, weight=self.distances[i][j])
                elif self.distances[i][j] != -1:
                    g.add_edge(i, j, weight=self.distances[i][j]*3)

        shortpath = nx.all_pairs_dijkstra_path_length(g, weight='weight')
        for a, p in shortpath:
            for b in p:
                dist[a][b] = p[b]
        return dist

    def pcch(self, i, j, x):
        marks = np.array([math.inf for k in range(self.number_of_cities)])
        marks[i] = 0
        s = (np.array([0 for k in range(self.number_of_cities)]))
        s[i] = 1  # because i is definitively marked
        n = i
        while (s[j] == 0):
            for k in range(self.number_of_cities):
                if (x[n][k] != 0 and (marks[k] > marks[n]+self.distances[n][k])):
                    marks[k] = marks[n]+self.distances[n][k]
            min = math.inf
            for k in range(self.number_of_cities):
                if (s[k] == 0):
                    if (marks[k] < min):
                        n = k
                        min = marks[k]
            if (min == math.inf):
                return math.inf
            s[n] = 1
        return marks[j]

    def f_gen(self, egal_bias, x):
        if (egal_bias == - 1):
            def f_egal(i, j):
                v = x[i][j]
                x[i][j] = 0
                x[j][i] = 0
                p = self.pcch(i, j, x)
                x[i][j] = v
                x[j][i] = v
                if (p == math.inf):
                    return math.inf
                # return p
                return p-self.distances[i][j]
            return f_egal

        def f_util(i, j):
            v = x[i][j]
            x[i][j] = 0
            x[j][i] = 0
            p = self.pcch(i, j, x)
            x[i][j] = v
            x[j][i] = v
            if (p == math.inf):
                return math.inf

            return self.commuter_matrix[i][j] * (p - self.distances[i][j])**egal_bias
        return f_util

    def object_function(self, egal_bias):
        if (egal_bias == -1):
            def f(dis: list, budget=1e9):
                a = max(max(dis))
                return a-1/(budget+1e5)
        else:
            def f(dis: List[List[float]], budget=1e9):
                ans = sum([float(dis[i][j])**egal_bias * self.commuter_matrix[i][j]
                          for i in range(len(dis)) for j in range(len(dis))])
                return ans-1/(budget+1e5)
        return f

    def filter_add(self, x_init=None, egal_bias=None, norm=0):
        if x_init is None:
            x_init = self.x.copy()
        if egal_bias == None:
            egal_bias = self.egal_bias
        gamma = []
        social_cost = self.of(self.compute_distance(
            np.where(x_init == 1, 1, 0)), self.evaluate_budget(x_init))
        of = self.object_function(egal_bias)
        for i in range(0, self.number_of_cities):
            for j in range(i+1, self.number_of_cities):
                if x_init[i][j] == -1:
                    x_init[i][j] = x_init[j][i] = 1
                    new_social_cost = of(
                        self.compute_distance(np.where(x_init == 1, 1, 0)), self.evaluate_budget(x_init))
                    x_init[i][j] = x_init[j][i] = -1
                    gamma.append((new_social_cost-social_cost) /
                                 self.distances[i][j]**norm)
        gamma_ord = np.argsort(np.array(gamma))
        indexes_source = np.array([None for i in range(len(gamma))])
        t = 0
        for i in range(self.number_of_cities):
            for j in range(i+1, self.number_of_cities):
                if x_init[i][j] == -1:
                    indexes_source[t] = (i, j)
                    t += 1
        gamma_ord_indexes = deque([None for i in range(len(gamma))])
        t = 0
        for i in range(self.number_of_cities):
            for j in range(i+1, self.number_of_cities):
                if x_init[i][j] == -1:
                    gamma_ord_indexes[t] = indexes_source[gamma_ord[t]]
                    t += 1

        # for i in range(1):
        #     x_init[gamma_ord_indexes[i][0]][gamma_ord_indexes[i][1]] = 1
        #     x_init[gamma_ord_indexes[i][1]][gamma_ord_indexes[i][0]] = 1
        return gamma_ord_indexes

    def filter_delete(self, x_init=None, egal_bias=None, norm=0):
        if x_init is None:
            x_init = self.x.copy()
        if egal_bias == None:
            egal_bias = self.egal_bias
        gamma = []
        social_cost = self.of(self.compute_distance(
            abs(x_init)), self.evaluate_budget(x_init))
        of = self.object_function(egal_bias)
        for i in range(0, self.number_of_cities):
            for j in range(i+1, self.number_of_cities):
                if x_init[i][j] == -1:
                    x_init[i][j] = x_init[j][i] = 0
                    new_social_cost = of(
                        self.compute_distance(abs(x_init)), self.evaluate_budget(x_init))
                    x_init[i][j] = x_init[j][i] = -1
                    gamma.append((new_social_cost-social_cost) /
                                 self.distances[i][j]**norm)
        gamma_ord = np.argsort(np.array(gamma))
        indexes_source = np.array([None for i in range(len(gamma))])
        t = 0
        for i in range(self.number_of_cities):
            for j in range(i+1, self.number_of_cities):
                if x_init[i][j] == -1:
                    indexes_source[t] = (i, j)
                    t += 1
        gamma_ord_indexes = deque([None for i in range(len(gamma))])
        t = 0
        for i in range(self.number_of_cities):
            for j in range(i+1, self.number_of_cities):
                if x_init[i][j] == -1:
                    gamma_ord_indexes[t] = indexes_source[gamma_ord[t]]
                    t += 1

        return gamma_ord_indexes

    def is_maximal(self, x, b):
        pos = x == -1
        for i in range(len(pos)):
            if b+self.distances[pos[i][0]][pos[i][1]] < self.budget:
                return False
        return True

    def branch(self, x, depth, min_cost, max_cost, social_cost):

        self.cntBranch += 1
        if (min_cost > self.budget):
            return
        if (max_cost > self.budget):
            if len(self.edge_in_order) <= depth:
                return
            if social_cost is None:
                xp = abs(x)
                social_cost = self.of(self.compute_distance(xp), max_cost)
            if social_cost >= self.best_social_cost:  # early stopping
                return
            x[self.edge_in_order[depth][0]][self.edge_in_order[depth][1]] = 0
            x[self.edge_in_order[depth][1]][self.edge_in_order[depth][0]] = 0
            # if self.is_connected(abs(x)): # not necessary to be connected
            self.branch(x, depth+1,
                        min_cost=min_cost,
                        max_cost=max_cost -
                        self.distances[self.edge_in_order[depth]
                                       [0]][self.edge_in_order[depth][1]],
                        social_cost=None)
            x[self.edge_in_order[depth][0]][self.edge_in_order[depth][1]] = 1
            x[self.edge_in_order[depth][1]][self.edge_in_order[depth][0]] = 1
            self.branch(
                x, depth+1,
                min_cost=min_cost +
                self.distances[self.edge_in_order[depth]
                               [0]][self.edge_in_order[depth][1]],
                max_cost=max_cost,
                social_cost=social_cost)
            x[self.edge_in_order[depth][0]][self.edge_in_order[depth][1]] = -1
            x[self.edge_in_order[depth][1]][self.edge_in_order[depth][0]] = -1
        else:
            xp = abs(x)
            if social_cost is None:
                social_cost = self.of(self.compute_distance(xp), max_cost)
            if social_cost < self.best_social_cost:
                self.best_network = xp
                self.best_social_cost = social_cost

    def mst(self):

        g = nx.Graph()
        for i in range(self.number_of_cities):
            g.add_node(i)
            for j in range(i+1, self.number_of_cities):
                if self.x[i][j] != 0:
                    g.add_edge(i, j, weight=self.distances[i][j])
        mst = nx.minimum_spanning_tree(g)
        mst_size = 0
        for i, j in mst.edges:
            mst_size += self.distances[i][j]
        return mst_size, mst

    def add_edges_iterative(self, num_add, egal_biases: List[int]):
        edges = []
        while True:
            if num_add < 1:
                return edges
            num_add -= 1

            mats = [None for _ in egal_biases]
            sorts = [None for _ in egal_biases]
            for i in range(len(egal_biases)):
                sorts[i] = self.filter_add(
                    mats[i], egal_bias=egal_biases[i], norm=1)
            rank = {sorts[0][i]: 0 for i in range(len(sorts[0]))}
            for i in range(len(egal_biases)):
                for j in range(len(sorts[i])):
                    rank[sorts[i][j]] = max(rank[sorts[i][j]], j)
            rank = sorted(rank.items(), key=lambda x: x[1])
            self.x[rank[0][0][0], rank[0][0][1]] = 1
            self.x[rank[0][0][1], rank[0][0][0]] = 1
            edges.append(rank[0][0])

    def delete_edges_iterative(self, num_delete, egal_biases: List[int]):
        edges = []
        while True:
            if num_delete == -1:
                if self.evaluate_budget(abs(self.x)) <= self.budget:
                    return edges
            else:
                if num_delete < 1:
                    return edges
                num_delete -= 1

            mats = [None for _ in egal_biases]
            sorts = [None for _ in egal_biases]
            for i in range(len(egal_biases)):
                sorts[i] = self.filter_delete(
                    mats[i], egal_bias=egal_biases[i], norm=1)
            rank = {sorts[0][i]: 0 for i in range(len(sorts[0]))}
            for i in range(len(egal_biases)):
                for j in range(len(sorts[i])):
                    rank[sorts[i][j]] = max(rank[sorts[i][j]], j)
            rank = sorted(rank.items(), key=lambda x: x[1])
            self.x[rank[0][0][0], rank[0][0][1]] = 0
            self.x[rank[0][0][1], rank[0][0][0]] = 0
            edges = [rank[0][0]]+edges

    def run_local_search(self, timeout, x_init=None, num_del=2, num_add=2, time_before=0):
        beginning = time.time()
        if x_init is None:
            res = self.delete_edges_iterative(-1, [self.egal_bias])
        else:
            self.x = x_init
        x = abs(self.x)
        cost = self.evaluate_budget(x)
        assert (cost <= self.budget)

        def subsets(n, m, currset, ans):
            ans.append(currset)
            if m == 0:
                return
            for i in range(currset[-1]+1 if len(currset) else 0, n):
                subsets(n, m-1, currset+[i], ans)
        ein, eout = [], []
        for i in range(self.number_of_cities):
            for j in range(i+1, self.number_of_cities):
                if self.distances[i][j] > 0:
                    if x[i][j] == 1:
                        ein.append((i, j))
                    else:
                        eout.append((i, j))

        def add_edge(edge):
            nonlocal cost, ein, eout, x
            x[edge[0]][edge[1]] = x[edge[1]][edge[0]] = 1
            cost += self.distances[edge[0]][edge[1]]
            ein.append(edge)
            eout.remove(edge)

        def del_edge(edge):
            nonlocal cost, ein, eout, x
            x[edge[0]][edge[1]] = x[edge[1]][edge[0]] = 0
            cost -= self.distances[edge[0]][edge[1]]
            ein.remove(edge)
            eout.append(edge)
        while True:
            if time.time()-beginning+time_before > timeout:
                print('Time out!', file=sys.stderr)
                break
            social_cost = self.of(self.compute_distance(x), cost)
            flag = False
            del_sets = []
            subsets(len(ein), num_del, [],  del_sets)
            for i in range(len(del_sets)):
                del_sets[i] = [ein[j] for j in del_sets[i]]

            add_sets = []
            subsets(len(eout), num_add, [], add_sets)
            for i in range(len(add_sets)):
                add_sets[i] = [eout[j] for j in add_sets[i]]
            delete_add_pairs = []
            for i in range(len(del_sets)):
                for j in range(len(add_sets)):
                    delete_add_pairs.append((del_sets[i], add_sets[j]))
            del_sets.sort(key=lambda x: sum(
                [self.distances[i][j] for i, j in x]))
            add_sets.sort(key=lambda x: -sum(
                [self.distances[i][j] for i, j in x]))
            start = len(add_sets)
            for del_set in del_sets:
                if time.time()-beginning+time_before > timeout:
                    break
                del_cost = sum([self.distances[i][j] for i, j in del_set])
                while start > 0 and cost-del_cost+sum([self.distances[i][j] for i, j in add_sets[start-1]]) <= self.budget:
                    start -= 1
                for ith in range(start, len(add_sets)):
                    add_set = add_sets[ith]
                    for i, j in del_set:
                        del_edge((i, j))
                    for i, j in add_set:
                        add_edge((i, j))
                    extra = []
                    for (a, b) in random.sample(eout, len(eout)):
                        if self.distances[a][b] + cost <= self.budget:
                            extra.append((a, b))
                            add_edge((a, b))
                    if self.of(self.compute_distance(x)) < social_cost:
                        flag = True
                        break
                    for (a, b) in extra:
                        del_edge((a, b))
                    for i, j in add_set:
                        del_edge((i, j))
                    for i, j in del_set:
                        add_edge((i, j))
                if flag:
                    break
            if not flag:
                break
        self.best_social_cost = self.of(self.compute_distance(x), cost)
        self.best_network = copy.deepcopy(x)
        assert (self.evaluate_budget(x) <= self.budget)
        return x, time.time()-beginning+time_before

    def run(self, timeout: float = 1e9):

        self.best_social_cost = math.inf
        self.best_network = None
        self.cntBranch = 0

        if self.algo == 'local_search':
            x, time_cost = self.run_local_search(
                x_init=self.init_solution, num_del=1, num_add=2, time_before=0, timeout=timeout)
            if time_cost < timeout:
                x, time_cost = self.run_local_search(
                    x_init=x, num_del=2, num_add=2, time_before=time_cost, timeout=timeout)

            print('LS', self.budget, self.best_social_cost,
                  round(time_cost, 3), file=sys.stderr)
            return x, time_cost

        gamma = []
        x_init = self.x.copy()
        f = self.f_gen(self.egal_bias, x_init)
        for i in range(0, self.number_of_cities-1):
            for j in range(i+1, self.number_of_cities):
                if x_init[i][j] == -1:
                    gamma.append(f(i, j)/self.distances[i][j])
        gamma_ord = np.argsort(np.array(gamma))
        # gamma_ord = np.flip(gamma_ord) #this would order them from best to worst
        indexes_source = np.array([None for i in range(len(gamma))])
        t = 0
        for i in range(self.number_of_cities):
            for j in range(i+1, self.number_of_cities):
                if x_init[i][j] == -1:
                    indexes_source[t] = (i, j)
                    t += 1
        edge_in_order = deque([None for i in range(len(gamma))])
        t = 0
        for i in range(self.number_of_cities):
            for j in range(i+1, self.number_of_cities):
                if x_init[i][j] == -1:
                    edge_in_order[t] = indexes_source[gamma_ord[t]]
                    t += 1

        # initializing the stack of solution spaces
        for i in range(len(edge_in_order)):
            if x_init[edge_in_order[i][0]][edge_in_order[i][1]] == -1:
                break
        beginning = time.time()

        while (x_init[edge_in_order[0][0]][edge_in_order[0][1]] == 0):
            edge_in_order.popleft()
        while (x_init[edge_in_order[-1][0]][edge_in_order[-1][1]] == 1):
            edge_in_order.pop()
        self.edge_in_order = edge_in_order

        self.branch(x_init, 0, 0, self.evaluate_budget(abs(x_init)), None)
        if self.best_network is None:
            print("no solution found", x_init, self.budget, file=sys.stderr)
            return None, time.time()-beginning
        # if self.algo == 'mixed':
        #     return self.run_local_search(self.best_network, time_before=time.time()-beginning)
        print('BR'if self.algo == 'branch'else 'MX', self.budget, self.cntFolyd, self.cntBranch, self.best_social_cost,
              round(time.time()-beginning, 3), file=sys.stderr)
        return self.best_network, time.time()-beginning
