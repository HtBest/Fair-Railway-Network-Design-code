from utils import add_edges_iterative, delete_edges_iterative, get_budgets, local_search
from solver import Solver
import network_design as nd
import graph as g
import functools
import sys
import time
import timeit
from collections import defaultdict
import copy
import math
import numpy as np
import clingo
asp = """% check graph is valid
:- distance(A,A,_).
:- trips(A,A,_).

:- distance(A,B,_), distance(B,A,_).
:- trips(A,B,_), trips(B,A,_).

% generate a railway network
{pick(A,B)} :- distance(A,B,_).
:- N < #sum{X,A,B:pick(A,B),distance(A,B,X)}, parameter(budget,N).

%#show spp/3.
%#show badEdge/3.
%#show parameter/2.
#show pick/2.
%#show totalTrips/1.
"""

asp_no_waste = """
:- #sum{X,A,B:pick(A,B),distance(A,B,X)} < N-I, parameter(budget,N), distance(C,D,I), not pick(C,D).
"""


def floydWarshall(vertices, edges, distance):
    sp = dict()
    n = len(vertices)
    for i in range(n):
        vi = vertices[i]
        for j in range(i+1, n):
            vj = vertices[j]
            edge = (vi, vj)
            if edge in edges:
                sp[edge] = distance[edge]
            elif edge in distance:
                assert (vj, vi) not in edges
                sp[edge] = 3*distance[edge]
            else:
                sp[edge] = nd.INF
    for k in range(n):
        vk = vertices[k]
        for i in range(k+1, n):
            vi = vertices[i]
            eik = sp[(vk, vi)]
            for j in range(i+1, n):
                vj = vertices[j]
                sp[(vi, vj)] = min(sp[(vi, vj)], eik+sp[(vk, vj)])
        for i in range(k):
            vi = vertices[i]
            eik = sp[(vi, vk)]
            for j in range(i+1, k):
                vj = vertices[j]
                sp[(vi, vj)] = min(sp[(vi, vj)], eik+sp[(vj, vk)])
            for j in range(k+1, n):
                vj = vertices[j]
                sp[(vi, vj)] = min(sp[(vi, vj)], eik+sp[(vk, vj)])
    return sp


def computeSocialCost(travel_time, demand, p):
    cumul = 0
    total = 0
    if p == -1:
        for trip in demand:
            cumul = max(cumul, travel_time[trip])
        return cumul
    else:
        for trip, d in demand.items():
            cumul += travel_time[trip] ** p * d
            total += d
        return cumul ** (1/p) / total ** (1/p)


class IncrementalAPSP:
    def __init__(self, graph, p):
        instance = nd.Instance(graph, budget=0, egal_bias=p, P=[p])
        self.vertices = sorted(list(graph["vertices"]))
        self.n = len(self.vertices)
        self.distance = {}
        for i in graph['distance']:
            for j in graph['distance'][i]:
                self.distance[(str(i), str(j))] = graph['distance'][i][j]
                self.distance[(str(j), str(i))] = graph['distance'][i][j]
        self.trips = instance.trips
        self.p = p
        self.reset()

    def reset(self):
        self.ready = False
        self.pick = set()

    def addEdges(self, edges):
        self.ready = False
        self.pick.update(edges)

    def removeEdges(self, edges):
        self.ready = False
        self.pick.difference_update(edges)

    def getValue(self):
        if not self.ready:
            self.travel_time = floydWarshall(
                self.vertices, self.pick, self.distance)
            self.value = computeSocialCost(
                self.travel_time, self.trips, self.p)
            self.ready = True
        return self.value


class SocialCost(clingo.Propagator):
    def __init__(self, G, p=1, target=None):
        self.target = target
        self.apsp = IncrementalAPSP(G, p)

    def init(self, init):
        self.apsp.reset()
        self.reverseMap = defaultdict(set)
        allEdges = []
        for atom in init.symbolic_atoms.by_signature("pick", 2):
            symb = atom.symbol
            assert symb.arguments[0].type == clingo.SymbolType.Function and symb.arguments[1].type == clingo.SymbolType.Function
            edge = (symb.arguments[0].name, symb.arguments[1].name)
            assert edge[0] < edge[1]
            soLite = init.solver_literal(init.symbolic_atoms[symb].literal)
            self.reverseMap[-soLite].add(edge)
            allEdges.append(edge)
            init.add_watch(-soLite)
        self.apsp.addEdges(allEdges)
        self.nonPicked = []
        self.calls = 0

    def propagate(self, ctl, changes):
        assert changes
        self.nonPicked.extend(changes)
        self.apsp.removeEdges(
            [edge for change in changes for edge in self.reverseMap[change]])
        value = self.apsp.getValue()
        self.calls += 1
        if self.target is not None and value >= self.target:
            if not ctl.add_nogood(self.nonPicked):
                ctl.propagate()

    def undo(self, solver_id, assign, changes):
        assert solver_id == 0
        self.nonPicked = [e for e in self.nonPicked if e not in changes]
        self.apsp.addEdges(
            [edge for change in changes for edge in self.reverseMap[change]])

    def on_model(self, model):
        self.value = self.apsp.getValue()
        self.target = self.value


class PickingOrderBUGGY(clingo.Propagator):
    def __init__(self, edges):
        self.order = edges

    def init(self, init):
        self.directMap = dict()
        self.reverseMap = dict()
        self.assigned = dict()
        for atom in init.symbolic_atoms.by_signature("pick", 2):
            symb = atom.symbol
            edge = (symb.arguments[0].name, symb.arguments[1].name)
            soLite = init.solver_literal(init.symbolic_atoms[symb].literal)
            self.directMap[edge] = soLite
            self.assigned[soLite] = False
            init.add_watch(soLite)
            init.add_watch(-soLite)
        self.progress = 0
        self.literalOrder = [self.directMap[e] for e in self.order]

    def propagate(self, ctl, changes):
        for c in changes:
            self.assigned[abs(c)] = True
        while self.progress < len(self.literalOrder) and self.assigned[self.literalOrder[self.progress]]:
            self.progress += 1
        # while self.progress < len(self.literalOrder):
        #     assert self.literalOrder[self.progress] in self.assigned
        #     if not self.assigned[self.literalOrder[self.progress]]:
        #         break
        #     self.progress += 1
        # print(f"prop {changes}, {self.assigned}, {self.progress}, {self.literalOrder}")
        print(f"prop {changes}, {self.progress}")

    def undo(self, solver_id, assign, changes):
        for c in changes:
            self.assigned[abs(c)] = False
        # while self.progress > 0:
        #     assert self.literalOrder[self.progress] in self.assigned
        #     if not self.assigned[self.literalOrder[self.progress]]:
        #         break
        #     self.progress -= 1
        while self.progress > 0 and self.assigned[self.literalOrder[self.progress-1]]:
            self.progress -= 1
        # print(f"undo {changes}, {self.assigned}, {self.progress}, {self.literalOrder}")
        print(f"undo {changes}, {self.progress}")

    def decide(self, solver_id, assignment, fallback):
        assert self.progress >= 0
        assert self.progress <= len(self.literalOrder)
        if self.progress == len(self.literalOrder):
            return 0
        e = self.literalOrder[self.progress]
        # print(f"decide {e}, {self.assigned}",flush=True)
        print(
            f"decide {e}, {self.progress}, {[(i, self.assigned[i]) for i in self.literalOrder]}", flush=True)
        assert e in self.assigned
        assert not self.assigned[abs(e)]
        return e


class PickingOrderSlow(clingo.Propagator):
    def __init__(self, edges):
        self.order = edges

    def init(self, init):
        self.directMap = dict()
        self.assigned = dict()
        for atom in init.symbolic_atoms.by_signature("pick", 2):
            symb = atom.symbol
            edge = (symb.arguments[0].name, symb.arguments[1].name)
            soLite = init.solver_literal(init.symbolic_atoms[symb].literal)
            self.directMap[edge] = soLite
            self.assigned[soLite] = False
            init.add_watch(soLite)
            init.add_watch(-soLite)
        self.literalOrder = [self.directMap[e] for e in self.order]

    def propagate(self, ctl, changes):
        for c in changes:
            self.assigned[abs(c)] = True

    def undo(self, solver_id, assign, changes):
        for c in changes:
            self.assigned[abs(c)] = False

    def decide(self, solver_id, assignment, fallback):
        for i in self.literalOrder:
            if not self.assigned[abs(i)]:
                return i
        return 0


@functools.cache
def getEdgeOrder(instance: nd.Instance, orderLinks):
    s = Solver(instance)
    edges = s.delete_edges_iterative(len(instance.edges), [instance.egal_bias])
    # nbEdges = len(H.to_dict()["edges"])
    # _, edges = delete_edges_iterative(H, nbEdges, [1]) if orderLinks else add_edges_iterative(H, nbEdges, [1])
    # edges = list(reversed(edges))
    edges = list(edges)
    # vertices = list(H.to_dict()["vertices"])
    vertices = list(instance.vertices)
    dedges = [(vertices[i], vertices[j]) for (i, j) in edges]
    dedges = [(min(i, j), max(i, j)) for (i, j) in dedges]
    return dedges


def timeBoundSolve(ctl, prop, timeLimit=-1, verbose=False):
    startTime = time.process_time()
    ctl.ground()
    totalCalls = 0
    iterations = 0
    model = set()
    with ctl.solve(yield_=True, async_=True, on_model=prop.on_model) as hnd:
        while True:
            hnd.resume()
            timeBudget = max(0, startTime+timeLimit -
                             time.process_time()) if timeLimit > 0 else None
            if not hnd.wait(timeBudget):
                timeout = True
                break
            totalCalls += prop.calls
            iterations += 1
            # print(totalCalls,time.process_time()-startTime,prop.value,m)
            if hnd.model() is None:
                timeout = False
                break
            model = copy.copy(prop.apsp.pick)
    # print(f"timeout {timeout}, exhaust {model}, bestVal {prop.value if model is not None else None}")
    # assert False
    if verbose:
        print(prop.calls, totalCalls, iterations, file=sys.stderr)
    return (timeout, model)


def solve(instance: nd.Instance, budget=3500, p=1, orderLinks=True, noWasteConstraint=True, timeLimit=-1, verbose=False):
    orderOption = ["--heuristic=Domain"] if orderLinks is not None else []
    ctl = clingo.Control(["0"] + orderOption)
    p = instance.egal_bias
    # instance = nd.Instance(instance.graph, budget=budget, egal_bias=p)
    ctl.add(asp)
    if noWasteConstraint:
        ctl.add(asp_no_waste)
    ctl.add(instance.toASP())
    if orderLinks is not None:
        ordered_edges = getEdgeOrder(instance, orderLinks)
        # print(getEdgeOrder(G,True))
        # print(getEdgeOrder(G,False))
        # assert False
        heuristics = [
            f"#heuristic pick({e[0]},{e[1]}). [{len(ordered_edges)-i},true]" for i, e in enumerate(ordered_edges)]
        order_heuristic = "\n".join(heuristics)
        # prop2 = PickingOrderSlow(ordered_edges)
        # ctl.register_propagator(prop2)
        ctl.add(order_heuristic)
    prop = SocialCost(instance.graph, p)
    ctl.register_propagator(prop)
    (timeout, model) = timeBoundSolve(ctl, prop, timeLimit, verbose)
    print("EX", instance.budget)
    if not model or timeout:
        return None
    else:
        vertices = prop.apsp.vertices
        distance = prop.apsp.distance
        # return model
        return (model, {p: computeSocialCost(floydWarshall(vertices, model, distance), instance.trips, p)})
        # return (model, { p : computeSocialCost(floydWarshall(vertices,model,distance),instance.trips,p) for p in [-1,1,2,3,6]})


# 301 -> solvingTime5
# 30  -> solvingTime3
def measureTime():
    maxTime = 60
    budgetSampling = 20  # 6 # 301
    maps = ["france"]
    maps = ["us", "france"]
    sizes = list(range(7, 12))  # [7,8]
    # sizes = list(range(10,11)) #[7,8]
    ps = [4]  # [1,4]
    samples = defaultdict(int)
    times = defaultdict(float)
    timedout = defaultdict(bool)
    for n in sizes:
        for country in maps:
            G = g.Graph()
            G.load("map_instances/" + country + ".city", n)
            getEdgeOrder(G, True)
            getEdgeOrder(G, False)
            budgets = get_budgets(G, budgetSampling)  # [0:2]
            for p in ps:
                for b in budgets:
                    for order in [True, False]:
                        #                    for order in [True,False,None]:
                        for noWaste in [True, False]:
                            allParams = (n, country, p, b, order, noWaste)
                            params = (n, country, p, order, noWaste)
                            if not timedout[params]:
                                (nb, time) = timeit.Timer(lambda: solve(
                                    G, budget=b, p=p, orderLinks=order, noWasteConstraint=noWaste, timeLimit=maxTime)).autorange()
                                print(
                                    f"callback {time: <2f}/{nb: <3}  {allParams} {params}", file=sys.stderr)
                                samples[params] += nb
                                times[params] += time
                                timedout[params] |= time > maxTime
                            else:
                                print(f"skip {allParams}", file=sys.stderr)
    means = {params: (times[params]/nb) for params, nb in samples.items()}
    organized = defaultdict(dict)
    for params, value in means.items():
        n = params[0]
        organized[n][params[1:]] = value if not timedout[params] else "nan"
    cols = organized[sizes[0]]
    print("n", end=" ")
    for col in cols:
        print(".".join(map(str, col)), end=" ")
    print()
    for n in sizes:
        print(n, end=" ")
        for col in cols:
            print(organized[n][col], end=" ")
        print()
    # print(organized)


def checkLocalSearch():
    budgetSampling = 300
    maps = ["france", "uk", "canada", "russia",
            "spain", "us", "germany", "brazil"]
    # maps = ["us",]
    # sizes = list(range(7,13))#[7,8,9,10,11,12]
    sizes = list(range(10, 11))
    ps = [1]  # ,4,10]
    betters = defaultdict(int)
    differs = defaultdict(int)
    ratios = defaultdict(int)
    samples = defaultdict(int)
    for country in maps:
        for n in sizes:
            G = g.Graph()
            G.load("map_instances/" + country + ".city", n)
            budgets = get_budgets(G, budgetSampling)  # [0:2]
            # print(len(budgets),file=sys.stderr)
            # assert False
            for p in ps:
                for b in budgets:
                    # answer = solve(G,budget=b,p=p)
                    answer = solve(G, budget=b, p=p, orderLinks=True,
                                   noWasteConstraint=True, timeLimit=-1)
                    assert answer is not None
                    # do not save result to files
                    model, norms = local_search(G, budget=b, p=p)
                    # allParams = (country,n,p,b)
                    # params = (country,n,p)
                    params = (n, p)
                    samples[params] += 1
                    if answer:
                        pmodel, pnorms = answer
                        differs[params] += norms[p] - pnorms[p]
                        if pnorms[p] == norms[p]:
                            print("Match", b, end=" ", file=sys.stderr)
                        elif pnorms[p] < norms[p]:
                            betters[params] += 1
                            ratios[params] += (norms[p] -
                                               pnorms[p]) / pnorms[p]
                            print("Better:", b,
                                  pnorms[p], norms[p], end=" ", file=sys.stderr)
                        else:
                            print("Error:", b, pnorms[p],
                                  norms[p], file=sys.stderr)
                            assert False
                print(samples, betters, ratios, differs, file=sys.stderr)
        print(samples, betters, ratios, differs, file=sys.stderr)
    organized = defaultdict(dict)
    for params, value in samples.items():
        n = params[0]
        organized[n][params[1:]] = (
            samples[params], betters[params], ratios[params]/samples[params])
    cols = organized[sizes[0]]
    print("n", end=" ")
    for col in cols:
        for t in ["samples", "betters", "ratios"]:
            print(".".join([t]+list(map(str, col))), end=" ")
    print()
    for n in sizes:
        print(n, end=" ")
        for col in cols:
            for data in organized[n][col]:
                print(data, end=" ")
        print()
