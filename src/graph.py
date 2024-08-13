from collections.abc import Iterable
import math
import networkx as nx
import numpy as np
import random
from math import sqrt
import json
import re
import sys


class Graph:
    def __init__(self, seed=None, filename=None):
        self.graph = nx.Graph()
        self.budget = -1
        self.egal_bias = 1
        self.rnd = random.Random(seed)
        if filename:
            self.load(filename)

    def add_vertex(self, n, x, y, size):
        self.graph.add_node(n, x=x, y=y, size=size)

    def add_edge(self, a, b, distance):
        self.graph.add_edge(a, b, distance=round(distance))

    def length(self, a, b):
        # dx = self.graph.nodes[a]['x'] - self.graph.nodes[b]['x']
        # dy = self.graph.nodes[a]['y'] - self.graph.nodes[b]['y']
        # ref: http://powerappsguide.com/blog/post/formulas-calculate-the-distance-between-2-points-longitude-latitude
        # ref: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
        lat1 = self.graph.nodes[a]['y']/18000*math.pi
        lon1 = self.graph.nodes[a]['x']/18000*math.pi
        lat2 = self.graph.nodes[b]['y']/18000*math.pi
        lon2 = self.graph.nodes[b]['x']/18000*math.pi

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        R = 6373.0

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        return distance

    def update_distance(self, a, b, value):
        if value < 0:
            value = 1e9
        if a > b:
            a, b = b, a
        if not self.graph.has_edge(a, b):
            self.graph.add_edge(a, b, distance=round(value))
        self.graph[a][b]['distance'] = round(value)
        if value >= 1e9:
            self.graph.remove_edge(a, b)

    def apply_matrix(self, matrix):
        for i, a in enumerate(self.graph.nodes):
            for j, b in enumerate(self.graph.nodes):
                if matrix[i][j] == 0 and a < b:
                    self.update_distance(a, b, 1e9)

    def load(self, filename, n=None):
        with open(filename, 'r') as file:
            line = file.readline()
            parts = re.split(r'[\s:]+', line.strip())
            if not parts:
                return
            realn = int(parts[0])
            n = realn if n is None else min(n, realn)
            m = int(parts[1]) if len(parts) > 1 else 0

            for i in range(realn):
                line = file.readline().strip()
                parts = re.split(r'[\s:]+', line)
                if not parts:
                    continue
                if i >= n:
                    continue
                name, x, y, size = parts[0].lower(), float(
                    parts[1]), float(parts[2]), int(parts[3])
                self.add_vertex(name, x, y, size)
            for i in self.graph.nodes():
                for j in self.graph.nodes():
                    if i < j:
                        dis = self.length(i, j)
                        self.update_distance(i, j, dis)
                        # print(i, j, dis)
            for _ in range(m):
                line = file.readline().strip()
                while line == "":  # Skip empty lines if any
                    line = file.readline().strip()
                parts = re.split(r'[\s:]+', line)
                if not parts:
                    continue

                a, b, dis = parts[0].lower(), parts[1].lower(), int(
                    parts[2]) if len(parts) > 2 else self.length(parts[0].lower(), parts[1].lower())
                if a not in self.graph.nodes or b not in self.graph.nodes:
                    print('Invalid edge:', a, b, 'skipped', file=sys.stderr)
                    continue
                self.update_distance(a, b, dis)

    def to_dict(self):
        vertices = {i: {'x': self.graph.nodes[i]['x'], 'y': self.graph.nodes[i]
                        ['y'], 'size': self.graph.nodes[i]['size']} for i in self.graph.nodes}

        edges = []
        for i, j in self.graph.edges():
            if i < j:
                edges.append((i, j))
            else:
                edges.append((j, i))
        distance = {}
        for i, j in self.graph.edges():
            if i < j:
                if i not in distance:
                    distance[i] = {}
                distance[i][j] = self.graph[i][j]['distance']
            else:
                if j not in distance:
                    distance[j] = {}
                distance[j][i] = self.graph[i][j]['distance']
        data = {'vertices': vertices, 'edges': edges, 'distance': distance}
        return data

    def mst(self):
        mst = nx.minimum_spanning_tree(self.graph, weight='distance')
        return sum([int(mst[a][b]['distance']) for a, b in mst.edges()])
