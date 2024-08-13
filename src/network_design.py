#! /usr/bin/env python3
"""TODO
"""

import math
import random
from collections import defaultdict
import numpy as np

INF = 2147483647


def return_inf():
    return INF


def floydWarshall(vertices, edges, distance):
    # print('edges:', edges)
    # print('distance:', distance)
    sp = defaultdict(return_inf)
    for i in vertices:
        sp[(i, i)] = 0
        for j in vertices:
            if (i, j) in edges:
                sp[(i, j)] = distance[(i, j)]
                sp[(j, i)] = distance[(i, j)]
            elif (i, j) in distance and (j, i) not in edges:
                sp[(i, j)] = 3*distance[(i, j)]
                sp[(j, i)] = 3*distance[(i, j)]
    for k in vertices:
        for i in vertices:
            for j in vertices:
                sum = min(sp[(i, k)]+sp[(k, j)], INF)
                sp[(i, j)] = min(sp[(i, j)], sum)
    return sp


class Instance:
    def __init__(self, graph=None, budget=0, egal_bias=0, algo='', P=[], mix_del=None, init_solution=None, json=None):
        if json is not None:
            self.graph = json['graph']
            self.budget = json['budget']
            self.egal_bias = json['egal_bias']
            self.P = json['P']
        else:
            self.graph = graph
            self.budget = budget
            self.egal_bias = egal_bias
            self.P = P
        self.mix_del = mix_del
        self.algo = algo
        self.init_solution = init_solution
        vertices = self.graph['vertices']
        edges = self.graph['edges']
        self.vertices = [str(V) for V in vertices.keys()]
        self.coordinates = {str(v): (data['x'], data['y'])
                            for (v, data) in vertices.items()}
        self.distance = {}
        for i in self.graph['distance']:
            for j in self.graph['distance'][i]:
                self.distance[(str(i), str(j))] = self.graph['distance'][i][j]
                self.distance[(str(j), str(i))] = self.graph['distance'][i][j]
        self.size = {str(V): data['size'] for (V, data) in vertices.items()}
        self.edges = [(str(U), str(V)) for (U, V) in edges]
#    def initialize(self):
        self.sp = floydWarshall(self.vertices, self.edges, self.distance)
        self.calcTrips()

    def n2norm(self, i, j):
        lat1 = self.coordinates[i][1]/18000*math.pi
        lon1 = self.coordinates[i][0]/18000*math.pi
        lat2 = self.coordinates[j][1]/18000*math.pi
        lon2 = self.coordinates[j][0]/18000*math.pi

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        R = 6373.0

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance

    def low_precision(self, num):
        if num < 10:
            return num
        return self.low_precision(round(num/10))*10

    def toASP(self, taget=0, precision=0):
        facts = []
        if precision:
            facts.append(f"target({taget}).")
            facts.append(f"precision({precision}).")
        facts.append(f"parameter(budget,{self.budget}).")
        if self.egal_bias > 0:
            facts.append(f"parameter(egal_bias,{self.egal_bias}).")
        for city in self.vertices:
            (x, y) = self.coordinates[city]
            facts.append(f"city({city}).")
            facts.append(f"size({city},{self.size[city]}).")
        for ((i, j), t) in self.trips.items():
            facts.append(f"trips({i},{j},{t}).")
            facts.append(
                f"shortest_path({i},{j},{self.sp[(i,j)]}).")
            if (i, j) in self.edges:
                facts.append(f"edge({i},{j}).")
                facts.append(
                    f"distance({i},{j},{self.distance[(i,j)]}).")
        return "\n".join(facts)

    def calcTripsA(self):  # check
        samples = 500
        sizeBias = 1
        population = [(C1, C2)
                      for C1 in self.vertices for C2 in self.vertices if C1 < C2]
        weights = [(self.size[C1]*self.size[C2])**sizeBias /
                   self.n2norm(C1, C2) for (C1, C2) in population]
        totalTrips = random.choices(population, weights, k=samples)
        self.trips = defaultdict(int)
        for trip in totalTrips:
            self.trips[trip] += 1

    def calcTrips(self):  # or this
        sizeBias = 1
        population = [(C1, C2)
                      for C1 in self.vertices for C2 in self.vertices if C1 < C2]
        weights = {(C1, C2): (self.size[C1]*self.size[C2]) **
                   sizeBias / self.n2norm(C1, C2) for (C1, C2) in population}
        max_weights = max(weights.values())
        weights = {k: 10000*v/max_weights for k, v in weights.items()}
        # print(weights)
        self.trips = defaultdict(int)
        for trip in population:
            self.trips[trip] = int(weights[trip])

    def to_dict(self):
        d = {"graph": self.graph, "budget": self.budget,
             "egal_bias": self.egal_bias, "P": self.P}
        return d

    def __str__(self):
        #        d = { "trips": self.trips, "sp": self.sp }
        d = {"trips": self.trips}
        return str(d)


# based on https://stackoverflow.com/a/49571213
def gini(incomes, weights):
    # The rest of the code requires numpy arrays.
    x = np.asarray(incomes)
    w = np.asarray(weights)
    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    sorted_w = w[sorted_indices]
    # Force float dtype to avoid overflows
    cumw = np.cumsum(sorted_w, dtype=float)
    cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
    return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (cumxw[-1] * cumw[-1]))


class Solution:
    def __init__(self, instance: Instance = None, atoms=None, matrix=None, time_used=0, json=None):
        if json is not None:
            self.instance = Instance(json=json['instance'])
            self.matrix = json['matrix']
            self.atoms = json['atoms']
            self.time = json['time']
            self.initialize()
            return
        self.instance = instance
        self.distance = instance.distance.copy()
        if matrix is None:
            self.atoms = atoms
            self.matrix = None
        else:
            self.atoms = None
            self.matrix = matrix.tolist() if type(matrix) == np.ndarray else matrix
        assert (self.atoms is not None) or (self.matrix is not None)
        self.time = time_used
        self.initialize()

    def initialize(self):
        collect = []
        self.pick = []
        self.paths = dict()
        if self.matrix is not None:
            for i, a in enumerate(self.instance.vertices):
                for j, b in enumerate(self.instance.vertices):
                    if i < j and self.matrix[i][j] == 1:
                        if a > b:
                            self.pick.append((b, a))
                            collect.append(f"pick({b},{a}).")
                        else:
                            self.pick.append((a, b))
                            collect.append(f"pick({a},{b}).")
        else:
            for atom in self.atoms:
                if atom.name == "pick":
                    assert (len(atom.arguments) == 2)
                    a = str(atom.arguments[0])
                    b = str(atom.arguments[1])
                    self.pick.append((a, b))
                else:
                    collect.append(str(atom))
        self.atoms = collect
        self.path = floydWarshall(
            self.instance.vertices, self.pick, self.instance.distance)
        self.cost = sum(self.instance.distance[edge] for edge in self.pick)
        self.travel_time = {
            trip: self.path[trip] for trip, _ in self.instance.trips.items()}
        self.gini = gini([self.travel_time[trip] for trip in self.travel_time.keys()], [
                         self.instance.trips[trip] for trip in self.travel_time.keys()])
        travel_distances = sorted(
            [(self.path[trip], self.instance.trips[trip]) for trip in self.travel_time.keys()], key=lambda x: x[0])
        total_trips = sum([self.instance.trips[trip]
                           for trip in self.instance.trips.keys()])
        top_10_dis = 0
        bottom_10_dis = 0
        cnt = 0
        for i in range(len(travel_distances)):
            cnt += travel_distances[i][1]
            if cnt >= total_trips*0.1:
                top_10_dis = travel_distances[i][0]
                break

        for i in range(len(travel_distances)):
            cnt += travel_distances[i][1]
            bottom_10_dis = travel_distances[i][0]
            if cnt > total_trips*0.9:
                break
        self.top_bottom_10_ratio = bottom_10_dis / top_10_dis
        self.max_degree = max([sum([1 for (i, j) in self.pick if i ==
                              vertex or j == vertex]) for vertex in self.instance.vertices])
        self.num_leaf = sum([1 for vertex in self.instance.vertices if sum(
            [1 for (i, j) in self.pick if i == vertex or j == vertex]) == 1])
        self.norms = {p: self.pnorm(p) for p in self.instance.P}
        cities = self.city_average_cost()
        best_city, worst_city = None, None
        for c in cities:
            if worst_city is None or cities[c]['average'] > worst_city['average']:
                worst_city = cities[c]
            if best_city is None or cities[c]['average'] < best_city['average']:
                best_city = cities[c]
        self.best_worst_city_ratio = worst_city['average'] / \
            best_city['average']

    def city_average_cost(self):
        cities = {v: {'average': 0, 'count': 0}
                  for v in self.instance.vertices}
        for trip, time in self.travel_time.items():
            cities[trip[0]]['average'] += time*self.instance.trips[trip]
            cities[trip[1]]['average'] += time*self.instance.trips[trip]
            cities[trip[0]]['count'] += self.instance.trips[trip]
            cities[trip[1]]['count'] += self.instance.trips[trip]

        for c in cities:
            if cities[c]['count'] > 0:
                cities[c]['average'] /= cities[c]['count']
        return cities

    def pnorm(self, p):
        cumul = 0
        total = 0
        if p == -1:
            for (trip, n) in self.travel_time.items():
                cumul = max(cumul, self.path[trip])
            return cumul
        for (trip, n) in self.travel_time.items():
            cumul += n ** p*self.instance.trips[trip]
            total += self.instance.trips[trip]
        return cumul ** (1/p) / total ** (1/p)

    def to_dict(self):
        d = {"instance": self.instance.to_dict(), "matrix": self.matrix,
             "atoms": None, "time": self.time}
        return d

    def __str__(self):
        paths = {(i, j): length for ((i, j), length)
                 in self.path.items() if (i < j)}
        d = {"cost": self.cost, "pick": self.pick,
             "path": paths, "norms": self.norms, "gini": self.gini, "time": self.time, "distance": self.distance}
        return str(d)
