#!/usr/bin/python3

from ortools.sat.python import cp_model

def makeGraph(edges: list[tuple[int, int]]):
    graph: dict[int, set[int]] = {}
    for edge in edges:
        if edge[0] not in graph:
            graph[edge[0]] = set()
        if edge[1] not in graph:
            graph[edge[1]] = set()
        graph[edge[0]].add(edge[1])
        graph[edge[1]].add(edge[0])
    # sort according to degree
    new_graph = {}
    for node in sorted(graph, key=lambda node: len(graph[node]), reverse=True):
        new_graph[node] = graph[node]
    return new_graph

def trivial_solve(node_count, edges):
    return node_count, 0, range(0, node_count)

def greedy_solve(node_count: int, edges: list[tuple[int, int]]):
    graph = makeGraph(edges)
    optimal = 0
    solution = [-1]*node_count
    for node in graph:
        neighbor_colors = set([solution[neighbor] for neighbor in graph[node] if solution[neighbor] != -1])
        color = 0
        while color in neighbor_colors:
            color += 1
        solution[node] = color
    colors = max(solution) + 1
    return colors, optimal, solution

def augmented_solve(colors: int, node_count: int, edges: list[tuple[int, int]]):
    model = cp_model.CpModel()
    variables = [model.new_int_var(0, min(i, colors-1), f'node{i}') for i in range(node_count)]
    for edge in edges:
        model.add(variables[edge[0]] != variables[edge[1]])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        optimal = 1 if status == cp_model.OPTIMAL else 0
        values = [solver.Value(variable) for variable in variables]
        return max(values) + 1, optimal, values

def constrained_programming_solve(node_count: int, edges: list[tuple[int, int]]):
    result = greedy_solve(node_count, edges)
    colors = result[0]
    while True:
        maybe_solution = augmented_solve(colors - 1, node_count, edges)
        if maybe_solution is not None:
            result = maybe_solution
            colors = result[0]
        else:
            break
    return result

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # build a trivial solution
    # every node has its own color
    colors, optimal, solution = constrained_programming_solve(node_count, edges)

    # prepare the solution in the specified output format
    output_data = str(colors) + ' ' + str(optimal) + '\n'
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

