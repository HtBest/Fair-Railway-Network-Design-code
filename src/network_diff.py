#! /usr/bin/env python3

import sys

from test_lib import load_graph


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {} <graph_file>'.format(sys.argv[0]))
        sys.exit(1)
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    graph1 = load_graph(file1)
    graph2 = load_graph(file2)
    cost1 = graph1.cost
    cost2 = graph2.cost
    print('Cost1:', cost1)
    print('Cost2:', cost2)
    for (i, j) in graph1.pick:
        if (i, j) not in graph2.pick:
            print('+ ({},{})'.format(i, j))
    for (i, j) in graph2.pick:
        if (i, j) not in graph1.pick:
            print('- ({},{})'.format(i, j))
