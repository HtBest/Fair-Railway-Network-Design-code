import json
import sys
import time
from typing import List
import matplotlib
from multiprocessing import Pool, cpu_count
import pathlib
import propagatorSolver as ps

import network_design as nd

from solver import Solver

WORKING_DIR = pathlib.Path(__file__).parent.parent.absolute()
SOLVER_DIR = WORKING_DIR / 'src' / 'solver'
OUTPUT_DIR = WORKING_DIR / 'tmp_output'

mypython = str(SOLVER_DIR / 'mypython.lp')


class Wrapper:
    def __init__(self) -> None:
        self.figsize = (8, 6)
        # self.figsize = (3.5, 2.5)
        self.seeds = [(998244353*i) % 1000000007 for i in range(10000)]

    def eval_one(self, instance: nd.Instance, params: dict) -> nd.Solution:

        # python solver
        if instance.algo == 'local_search':
            solver = Solver(instance)
            res1, tm = solver.run(**params)
            if res1 is None:
                return None
            sol = nd.Solution(instance, matrix=res1, time_used=tm)
        else:
            time_start = time.time()
            edges = ps.solve(instance, verbose=False)[0]
            mat = [[0 for _ in range(len(instance.vertices))]
                   for _ in range(len(instance.vertices))]
            toid = {v: i for i, v in enumerate(instance.vertices)}
            for (i, j) in edges:
                mat[toid[i]][toid[j]] = 1
                mat[toid[j]][toid[i]] = 1
            sol = nd.Solution(instance, matrix=mat,
                              time_used=time.time()-time_start)
        #####

        # ASP solver
        # program = str(SOLVER_DIR / 'utilitarian5.lp')
        # atoms, cost, time_used = clingoInterface.run_solver(
        #     [mypython, program], instance.toASP())
        # sol = nd.Solution(instance, atoms=atoms,time_used= time_used)
        #####
        # sol.initialize()
        return sol

    def eval(self, instance_groups: List[List[nd.Solution]], params: dict = {}) -> List[List[nd.Solution]]:
        max_processes = 1
        max_processes = cpu_count()-2
        print('max_processes:', max_processes, file=sys.stderr)
        with Pool(max_processes) as pool:
            poolapply = pool.apply_async

            def unfold(x):
                if type(x) == list:
                    return [unfold(y) for y in x]
                elif type(x) == dict:
                    return {key: unfold(value) for key, value in x.items()}
                else:
                    return poolapply(self.eval_one, args=(x, params))

            def fold(x):
                if type(x) == list:
                    return [fold(y) for y in x]
                elif type(x) == dict:
                    return {key: fold(value) for key, value in x.items()}
                else:
                    return x.get()
            data = unfold(instance_groups)

            data = fold(data)
            return data


def load_graph(file: str) -> nd.Solution:
    with open(file) as f:
        graph = json.load(f)
        graph['instance']['P'] = [1, 2, 3,4, 6, 10]
        graph = nd.Solution(json=graph)
        return graph
