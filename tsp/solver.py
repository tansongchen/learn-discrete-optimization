#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from random import random, shuffle, randint
from util import Point, length, visualize_solution
from sa import SA, SimpleSA

def objective_value(solution: list[int], dist: list[list[float]]):
    nodeCount = len(solution)
    obj = dist[solution[-1], solution[0]]
    for index in range(nodeCount-1):
        obj += dist[solution[index], solution[index+1]]
    return obj

def trivial(points: list[Point]):
    nodeCount = len(points)
    solution = range(0, nodeCount)
    obj = 0
    for i in range(nodeCount):
        obj += length(points[solution[i]], points[solution[(i+1) % nodeCount]])
    return solution, obj

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    if nodeCount < 10000:
        sa = SA(points)
        solution, obj = sa.solve()
    else:
        sa = SimpleSA(points)
        solution, obj = sa.solve()

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    # visualize_solution(points, solution)

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

