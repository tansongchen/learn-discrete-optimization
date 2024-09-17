#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from math import exp, sqrt
from random import random, randint, choice
from numpy import zeros, float64, int32, ndarray, concatenate, atan2

Point = namedtuple("Point", ["x", "y"])
Customer = namedtuple("Customer", ['index', 'demand', "position"])

def length(a: Point, b: Point) -> float:
    dx = (a.x - b.x)
    dy = (a.y - b.y)
    return sqrt(dx * dx + dy * dy)

def length2(a: Point, b: Point) -> float:
    dx = (a.x - b.x)
    dy = (a.y - b.y)
    return dx * dx + dy * dy

class SA:
    def __init__(self, depot: Point, customers: list[Customer], vehicle_count: int, vehicle_capacity: int):
        self.customers = customers
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.nodes = nodes = len(customers) + 1
        self.dist = dist = zeros((nodes, nodes), dtype=float64)
        positions = [depot] + [customer.position for customer in customers]
        for i in range(nodes):
            for j in range(i, nodes):
                dist[i, j] = dist[j, i] = length(positions[i], positions[j])
        self.good_neighbors: dict[int, set[int]] = {}
        for i in range(nodes):
            mean = self.dist[i, :].mean()
            self.good_neighbors[i] = set(j for j in range(nodes) if i != j and self.dist[i, j] < 2 * mean)

    def residual_capacity(self, solution: list[list[int]]):
        res = [self.vehicle_capacity] * len(solution)
        for i, tour in enumerate(solution):
            for j in tour[1:]:
                res[i] -= self.customers[j - 1].demand
        return res

    def objective_value(self, solution: list[list[int]]):
        obj = 0.0
        for tour in solution:
            obj += self.dist[tour[-1], tour[0]]
            for index in range(len(tour)-1):
                obj += self.dist[tour[index], tour[index+1]]
        return obj

    def build_greedy_solution(self) -> ndarray[int32]:
        start = randint(0, self.nodes-1)
        # at each step, choose the nearest neighbor
        solution = zeros(self.nodes, dtype=int32)
        solution[0] = start
        visited = {start}
        for i in range(1, self.nodes):
            last = solution[i-1]
            nearest = min([j for j in range(self.nodes) if j not in visited], key=lambda j: self.dist[last, j])
            solution[i] = nearest
            visited.add(nearest)
        return solution

    def temperature_heuristic(self):
        mean = self.dist.sum() / self.nodes / self.nodes
        return mean / 10, mean / 10000, 1000000

    def two_opt_neighbor(self, solution: ndarray[int32], old_obj: float):
        dist = self.dist
        nodes = len(solution)
        i = randint(0, nodes - 3)
        j = randint(i + 1, nodes - 1)
        from1, to1 = solution[i], solution[i+1]
        from2, to2 = solution[j], solution[(j+1) % nodes]

        new_solution = solution.copy()
        new_solution[i+1:j+1] = solution[i+1:j+1][::-1]
        new_obj = old_obj + dist[from1, from2] + dist[to1, to2] - dist[from1, to1] - dist[from2, to2]
        return new_solution, new_obj

    def three_opt_neighbor(self, solution: ndarray[int32], old_obj: float):
        dist = self.dist
        nodes = len(solution)
        i = randint(0, nodes - 5)
        j = randint(i + 1, nodes - 3)
        k = randint(j + 1, nodes - 1)
        from1, to1 = solution[i], solution[i+1]
        from2, to2 = solution[j], solution[j+1]
        from3, to3 = solution[k], solution[(k+1) % nodes]

        new_solution = solution.copy()
        block1 = solution[i+1:j+1]
        block2 = solution[j+1:k+1]
        new_solution[i+1:k+1] = block2[::-1] + block1
        new_obj = old_obj + dist[from1, from3] + dist[to2, to1] + dist[from2, to3] - dist[from1, to1] - dist[from2, to2] - dist[from3, to3]
        return new_solution, new_obj

    def two_opt_greedy(self, solution: ndarray[int32], obj: float):
        target = { 574: 40000, 1889: 378069 }
        while True:
            best_solution = solution.copy()
            best_obj = obj
            for i in range(0, self.nodes - 2):
                for j in range(i + 1, self.nodes):
                    from1, to1 = solution[i], solution[i+1]
                    from2, to2 = solution[j], solution[(j+1) % self.nodes]
                    new_obj = obj + self.dist[from1, from2] + self.dist[to1, to2] - self.dist[from1, to1] - self.dist[from2, to2]
                    if new_obj + 1e-10 < best_obj:
                        new_solution = solution.copy()
                        new_solution[i+1:j+1] = solution[i+1:j+1][::-1]
                        best_solution = new_solution
                        best_obj = new_obj
            if best_obj < obj:
                solution = best_solution
                obj = best_obj
                if obj < target.get(self.nodes, 0):
                    break
            else:
                break
        return solution, obj

    def tsp_neighbor(self, solution: ndarray[int32], obj: float):
        tourIndex = randint(0, len(solution) - 1)
        tour = solution[tourIndex]
        if len(tour) >= 6:
            new_tour, new_obj = self.three_opt_neighbor(tour, obj)
        elif len(tour) >= 4:
            new_tour, new_obj = self.two_opt_neighbor(tour, obj)
        else:
            new_tour = tour.copy()
            new_obj = obj
        new_solution = solution.copy()
        new_solution[tourIndex] = new_tour
        return new_solution, new_obj

    def reassign_neighbor(self, solution: list[list[int]], obj: float, res: list[int]):
        tourIndex = randint(0, len(solution) - 1)
        tour = solution[tourIndex].copy()
        if len(tour) == 1:
            return solution, obj, res
        customerIndex = randint(1, len(tour) - 1)
        customer = tour[customerIndex]
        customerDemand = self.customers[customer].demand
        possibleTours = [i for i in range(len(solution)) if i != tourIndex and res[i] >= customerDemand]
        if len(possibleTours) == 0:
            return solution, obj, res
        newTourIndex = choice(possibleTours)
        newTour = solution[newTourIndex].copy()
        newCustomerIndex = randint(1, len(newTour))
        newTour.insert(newCustomerIndex, customer)
        new_obj = obj \
            - self.dist[tour[customerIndex-1], customer] - self.dist[customer, tour[(customerIndex+1) % len(tour)]] \
            + self.dist[newTour[newCustomerIndex-1], customer] + self.dist[customer, newTour[(newCustomerIndex+1) % len(newTour)]]
        tour.pop(customerIndex)
        new_solution = solution.copy()
        new_solution[tourIndex] = tour
        new_solution[newTourIndex] = newTour
        assert self.objective_value(new_solution) == new_obj
        new_res = res.copy()
        new_res[tourIndex] += customerDemand
        new_res[newTourIndex] -= customerDemand
        return new_solution, new_obj, new_res

    def solve(self, solution: list[list[int]]):
        obj = self.objective_value(solution)
        res = self.residual_capacity(solution)
        best_solution = solution.copy()
        best_obj = obj
        t_max, t_min, steps = self.temperature_heuristic()
        for step in range(steps):
            t = t_max * (t_min / t_max) ** (step / steps)
            if random() < 2: # TSP move
                new_solution, new_obj = self.tsp_neighbor(solution, obj)
                new_res = res
            else: # reassign
                new_solution, new_obj, new_res = self.reassign_neighbor(solution, obj, res)
            accepted = new_obj < best_obj or random() < exp((best_obj - new_obj) / t)
            if accepted:
                solution = new_solution
                obj = new_obj
                res = new_res
            if obj < best_obj:
                best_solution = solution
                best_obj = obj
        # best_solution, best_obj = self.two_opt_greedy(best_solution, best_obj)
        return best_solution, best_obj

def trivial_solution(depot: Point, customers: list[Customer], vehicle_count: int, vehicle_capacity: int):
    vehicle_tours: list[list[int]] = []
    remaining_customers = set(customers)
    customer_count = len(customers)

    positions = [depot] + [customer.position for customer in customers]
    
    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([0])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: atan2(customer.position.y - depot.y, customer.position.x - depot.x))
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer.index)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used

    # checks that the number of customers served is correct
    assert sum([len(v) - 1 for v in vehicle_tours]) == len(customers)

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = 0.0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            for i in range(0, len(vehicle_tour)-1):
                obj += length(positions[vehicle_tour[i]],positions[vehicle_tour[i+1]])
            obj += length(positions[vehicle_tour[-1]],depot)

    return vehicle_tours, obj

def sa_solution(depot, customers, vehicle_count, vehicle_capacity):
    init_tours, _ = trivial_solution(depot, customers, vehicle_count, vehicle_capacity)
    sa = SA(depot, customers, vehicle_count, vehicle_capacity)
    best_tours, best_obj = sa.solve(init_tours)
    return best_tours, best_obj

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        position = Point(float(parts[1]), float(parts[2]))
        customers.append(Customer(i-1, int(parts[0]), position))

    #the depot is always the first customer in the input
    depot = customers[0].position
    vehicle_tours, obj = sa_solution(depot, customers[1:], vehicle_count, vehicle_capacity)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += ' '.join([str(customer) for customer in vehicle_tours[v]]) + ' ' + str(0) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

