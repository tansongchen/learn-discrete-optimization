#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from random import random, shuffle
from collections import namedtuple
from numpy import matrix, zeros, float32, arange, int32, ndarray
from turtle import Turtle, Screen, done, screensize

Point = namedtuple("Point", ['x', 'y'])

def length(point1: Point, point2: Point):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def visualize_solution(points: list[Point], solution: list[int]):
    coordinates: list[tuple[float, float]] = [(a.x, a.y) for a in points]
    screensize(800, 600, 'white')
    t = Turtle()
    t.goto(coordinates[0])
    t.down()
    for point in coordinates[1:]:
        t.goto(point)
    t.goto(coordinates[0])
    done()

def distance_matrix(points: list[Point]):
    nodeCount = len(points)
    dist = zeros((nodeCount, nodeCount), dtype=float32)
    for i in range(nodeCount):
        for j in range(i, nodeCount):
            dist[i, j] = dist[j, i] = length(points[i], points[j])
    return dist

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

def two_opt_neighbor(solution: ndarray[int32], old_obj: float, dist: list[list[float]]):
    nodeCount = len(solution)
    while True:
        i, j = int(random() * nodeCount), int(random() * nodeCount)
        i, j = min(i, j), max(i, j)
        if i != 0 and i != j: break
    new_solution = solution.copy()
    new_solution[i:j+1] = solution[i:j+1][::-1]
    from1, to1 = solution[i-1], solution[i]
    from2, to2 = solution[j], solution[(j+1) % nodeCount]
    new_obj = old_obj + dist[from1, from2] + dist[to1, to2] - dist[from1, to1] - dist[from2, to2]
    return new_solution, new_obj

def simulated_annealing(points: list[Point]):
    nodeCount = len(points)
    dist = distance_matrix(points)
    mean = dist.sum() / nodeCount / nodeCount
    solution = arange(0, nodeCount, dtype=int32)
    shuffle(solution)
    obj = objective_value(solution, dist)
    best_solution = solution.copy()
    best_obj = obj
    t_max, t_min, steps = mean / 10, mean / 100, 1000000
    for step in range(steps):
        t = t_max * (t_min / t_max) ** (step / steps)
        new_solution, new_obj = two_opt_neighbor(solution, obj, dist)
        accepted = new_obj < best_obj or random() < math.exp((best_obj - new_obj) / t)
        if accepted:
            solution = new_solution
            obj = new_obj
        if obj < best_obj:
            best_solution = solution
            best_obj = obj
    return best_solution, best_obj

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
        solution, obj = simulated_annealing(points)
    else:
        solution, obj = trivial(points)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

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

