#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def greedy(items: list[Item], capacity):
    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    items.sort(key=lambda item: item.value / item.weight, reverse=True)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    
    return value, taken
    
def dynamic_programming_compressed(items: list[Item], capacity):
    status = [0 for i in range(capacity+1)]
    content = [[] for i in range(capacity + 1)]
    for index, item in enumerate(items):
        for cap in range(capacity, 0, -1):
            if item.weight <= cap and (status[cap - item.weight] + item.value > status[cap]):
                status[cap] = status[cap - item.weight] + item.value
                content[cap] = content[cap - item.weight][:]
                content[cap].append(item.index)
    value = status[-1]
    pack = set(content[-1])
    taken = [(1 if i in pack else 0) for i in range(len(items))]
    return value, taken

def dynamic_programming(items: list[Item], capacity):
    status = [[0 for i in range(capacity+1)] for j in range(len(items)+1)]
    for index, item in enumerate(items):
        previous = status[index]
        toUpdate = status[index + 1]
        for cap in range(1, capacity + 1):
            if item.weight <= cap:
                toUpdate[cap] = max(previous[cap], previous[cap - item.weight] + item.value)
            else:
                toUpdate[cap] = previous[cap]
    value = status[-1][-1]
    row = capacity
    taken = [0]*len(items)
    for j in range(len(items), 0, -1):
        if status[j - 1][row] != status[j][row]:
            taken[j - 1] = 1
            row -= items[j - 1].weight
    return value, taken

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    
    problem_size = len(items) * capacity
    if problem_size <= 1e8:
        value, taken = dynamic_programming_compressed(items, capacity)
    else:
        value, taken = greedy(items, capacity)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

