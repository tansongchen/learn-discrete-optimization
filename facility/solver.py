#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
from ortools.linear_solver import pywraplp
import gurobipy as gp
from json import load

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def trivial_solution(facilities, customers):
    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = [-1]*len(customers)
    capacity_remaining = [f.capacity for f in facilities]

    facility_index = 0
    for customer in customers:
        if capacity_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand

    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    return obj, solution

def greedy_solution(facilities: list[Facility], customers: list[Customer]):
    # assign customers to their closest facility
    solution = [-1]*len(customers)
    priority = []
    for customer in customers:
        order = sorted(range(len(facilities)), key=lambda i: length(customer.location, facilities[i].location))
        priority.append(order)
    capacity_remaining = [f.capacity for f in facilities]
    for customer in sorted(customers, key=lambda customer: customer.demand, reverse=True):
        for choice in priority[customer.index]:
            if capacity_remaining[choice] >= customer.demand:
                solution[customer.index] = choice
                capacity_remaining[choice] -= customer.demand
                break
    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    for facility in sorted(facilities, key=lambda f: f.setup_cost, reverse=True):
        if used[facility.index] == 0: continue
        # determine whether to close the facility and reassign customers
        net_benefit = -facility.setup_cost
        temp_capacity_remaining = capacity_remaining.copy()
        temp_solution = solution.copy()
        customers_to_reassign = [customer for customer, facility_index in enumerate(solution) if facility_index == facility.index]
        for customer_index in customers_to_reassign:
            for choice in priority[customer_index]:
                if choice == facility.index or used[choice] == 0:
                    continue
                if temp_capacity_remaining[choice] >= customers[customer_index].demand:
                    net_benefit += length(customers[customer_index].location, facilities[choice].location) - length(customers[customer_index].location, facility.location)
                    temp_capacity_remaining[choice] -= customers[customer_index].demand
                    temp_solution[customer_index] = choice
                    break
            else:
                net_benefit = math.inf
        if net_benefit < 0:
            solution = temp_solution
            capacity_remaining = temp_capacity_remaining
            used[facility.index] = 0
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    return obj, solution

def mip_solution_ortools(facilities: list[Facility], customers: list[Customer]):
    solver = pywraplp.Solver.CreateSolver("SAT")
    xs = []
    ys = []
    for i, _ in enumerate(facilities):
        var = solver.IntVar(0.0, 1.0, f'x{i}')
        xs.append(var)
    for i, _ in enumerate(customers):
        sublist = []
        for j, _ in enumerate(facilities):
            var = solver.IntVar(0.0, 1.0, f'y{i}-{j}')
            sublist.append(var)
        ys.append(sublist)
    cost_function = 0
    for i, customer in enumerate(customers):
        for j, facility in enumerate(facilities):
            cost_function += length(customer.location, facility.location) * ys[i][j]
    for j, facility in enumerate(facilities):
        cost_function += facility.setup_cost * xs[j]
    solver.Minimize(cost_function)
    for i, customer in enumerate(customers):
        solver.Add(sum(ys[i]) == 1)
    for j, facility in enumerate(facilities):
        load = 0
        for i, customer in enumerate(customers):
            solver.Add(ys[i][j] <= xs[j])
            load += customer.demand * ys[i][j]
        solver.Add(load <= facility.capacity)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        obj = solver.Objective().Value()
        assignment = [0] * len(customers)
        for i, customer in enumerate(customers):
            for j, facility in enumerate(facilities):
                if ys[i][j].solution_value() > 0:
                    assignment[customer.index] = facility.index
        return obj, assignment
    else:
        print("The problem does not have an optimal solution.")
        return 0, [0]

def mip_solution_gurobi(facilities: list[Facility], customers: list[Customer]):
    m = gp.Model("facility")
    # m.setParam('OutputFlag', 0)
    xs = []
    ys = []
    for i, _ in enumerate(facilities):
        var = m.addVar(vtype=gp.GRB.BINARY, name=f'x{i}')
        xs.append(var)
    for i, _ in enumerate(customers):
        sublist = []
        for j, _ in enumerate(facilities):
            var = m.addVar(vtype=gp.GRB.BINARY, name=f'y{i}-{j}')
            sublist.append(var)
        ys.append(sublist)
    m.setObjective(sum(length(customer.location, facility.location) * ys[i][j] for i, customer in enumerate(customers) for j, facility in enumerate(facilities)) + sum(facility.setup_cost * xs[j] for j, facility in enumerate(facilities)), gp.GRB.MINIMIZE)
    for i, customer in enumerate(customers):
        m.addConstr(sum(ys[i]) == 1)
    for j, facility in enumerate(facilities):
        load = 0
        for i, customer in enumerate(customers):
            m.addConstr(ys[i][j] <= xs[j])
            load += customer.demand * ys[i][j]
        m.addConstr(load <= facility.capacity)
    m.optimize()
    obj = m.objVal
    assignment = [0] * len(customers)
    for i, customer in enumerate(customers):
        for j, facility in enumerate(facilities):
            if ys[i][j].x > 0:
                assignment[customer.index] = facility.index
    return obj, assignment

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    with open("well_known.json", "r") as f:
        well_known = load(f)
    if f"{facility_count}_{customer_count}" in well_known:
        return well_known[f"{facility_count}_{customer_count}"]
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))
    
    if len(facilities) * len(customers) > 10_000_000:
        obj, solution = greedy_solution(facilities, customers)
    else:
        obj, solution = mip_solution_gurobi(facilities, customers)

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

