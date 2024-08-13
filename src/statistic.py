import os
import sys
from draw_plot import draw as draw_plot
from test_lib import load_graph
from utils import pname


def generate_csv(path):
    if path[-1] == '/':
        path = path[:-1]

    files = os.listdir(path)
    files = [f for f in files if f.endswith('.json')]

    # budgets,P,data
    budgets = set()
    P = set()
    for filename in files:
        if 'p=' in filename:
            budgets.add(int(filename.split(' ')[0]))
            P.add(int(filename.split(' ')[1].split('p=')[1].split('.json')[0]))
    budgets = list(budgets)
    budgets.sort()
    P = list(P)
    P.sort()
    data = []
    for b in budgets:
        data.append([])
        for p in P:
            data[-1].append(load_graph(os.path.join(path, f'{b} p={p}.json')))
    # plot
    keys = ['gini', 'top_bottom_10_ratio', 'max_degree',
            'num_leaf', 'time', 'best_worst_city_ratio']
    name = ['Gini Coefficient',  r'Top/Bottom 10% Ratio',
            'Max Degree', 'Number of Leaves', 'Time', 'Best/Worst City Ratio']
    for norm in data[0][0].norms:
        keys.append(('norms', norm))
        name.append('$p='+pname(norm)+'$-social cost')

    for ith, k in enumerate(keys):
        if type(k) == tuple:
            k, curve_p = k
            keyname = k+' p='+pname(curve_p)
            def get_value(d): return getattr(d, k)[curve_p]
        else:
            def get_value(d): return getattr(d, k)
            keyname = k
        # csv
        with open(path+'/'+keyname+'.csv', 'w') as f:
            f.write('Budget\t'+name[ith]+',')
            f.write(','.join(['p='+pname(p) for p in P]))
            f.write('\n')
            for i in range(len(data)):
                f.write(str(budgets[i])+',')
                f.write(','.join([str(get_value(d)) for d in data[i]]))
                f.write('\n')
        draw_plot(path+'/', keyname+'.csv')

    for ith, curve in enumerate(P):
        with open(path+'/curve p='+pname(curve)+'.csv', 'w') as f:
            f.write('Budget\t$p='+pname(curve)+'$-optimal network,')
            f.write(','.join(['$p='+pname(p)+'$' for p in P]))
            f.write('\n')
            for i in range(len(data)):
                f.write(str(budgets[i])+',')
                f.write(
                    ','.join([str(getattr(data[i][ith], 'norms')[P[j]]) for j in range(len(P))]))
                f.write('\n')
        draw_plot(path +
                  '/', 'curve p='+pname(curve)+'.csv')

    keys = ['centrality',  'population', 'travel_demand']

    def centrality(d, city):
        sum_dis = 0
        num_dis = 0
        for (i, j) in d.instance.distance:
            if i == city or j == city:
                sum_dis += d.instance.distance[(i, j)]
                num_dis += 1
        return sum_dis / num_dis

    def centrality2(d, city):
        sum_dis = 0
        num_dis = 0
        for (i, j) in d.instance.distance:
            if i == city or j == city:
                sum_dis += d.instance.distance[(i, j)] ** 2
                num_dis += 1
        return sum_dis / num_dis

    funcs = [centrality,
             lambda d, city: d.instance.size[city],  # population
             lambda d, city:  # travel demand
             sum(d.instance.trips[(city, j)] for j in d.instance.vertices if (city, j) in d.instance.trips) + \
             sum(d.instance.trips[(j, city)]
                 for j in d.instance.vertices if (j, city) in d.instance.trips)]
    instance = data[0][0].instance
    for i, k in enumerate(keys):
        csv = []
        for j, p in enumerate(P):
            res = []
            for l in range(len(instance.vertices)):
                sumcost = 0
                sumtrips = 0
                for m in range(len(data)):
                    for n in range(len(instance.vertices)):
                        trip = (data[m][j].instance.vertices[l],
                                data[m][j].instance.vertices[n])
                        if trip in data[m][j].instance.trips:
                            sumtrips += data[m][j].instance.trips[trip]
                            sumcost += data[m][j].instance.trips[trip] * \
                                data[m][j].path[trip]
                        trip = (data[m][j].instance.vertices[n],
                                data[m][j].instance.vertices[l])
                        if trip in data[m][j].instance.trips:
                            sumtrips += data[m][j].instance.trips[trip]
                            sumcost += data[m][j].instance.trips[trip] * \
                                data[m][j].path[trip]
                res.append((funcs[i](data[0][0], instance.vertices[l]),
                           sumcost/sumtrips))  # x and y
            res.sort(key=lambda x: x[0])
            csv.append(res)

        with open(path+'/'+k+'.csv', 'w') as f:
            f.write(k+'\tAverage Cost,')
            f.write(','.join(['p='+pname(p) for p in P]))
            f.write('\n')
            for i in range(len(csv[0])):
                f.write(str(csv[0][i][0])+',')
                f.write(','.join([str(csv[j][i][1]) for j in range(len(P))]))
                f.write('\n')
        draw_plot(path+'/', k+'.csv')


if __name__ == '__main__':
    for i in range(1, len(sys.argv)):
        generate_csv(sys.argv[i])
