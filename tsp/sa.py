from math import exp
from random import random, randint
from numpy import zeros, float64, int32, ndarray, concatenate
from util import length, length2, Point

class SimpleSA:
    def __init__(self, points: list[Point]):
        self.points = points
        self.nodes = len(points)
        return
    
    def objective_value(self, solution: list[int]):
        obj = 0
        for index in range(self.nodes):
            obj += length(self.points[solution[index]], self.points[solution[(index+1) % self.nodes]])
        return obj

    def build_greedy_solution(self) -> ndarray[int32]:
        # at each step, choose the nearest neighbor
        solution = zeros(self.nodes, dtype=int32)
        solution[0] = 0
        visited = { 0 }
        for i in range(1, self.nodes):
            last = solution[i-1]
            min_distance = float('inf')
            nearest = None
            point = self.points[last]
            for j in range(self.nodes):
                if j in visited:
                    continue
                distance = length2(point, self.points[j])
                if distance < min_distance:
                    min_distance = distance
                    nearest = j
            solution[i] = nearest
            visited.add(nearest)
        return solution

    def solve(self):
        solution = self.build_greedy_solution()
        return solution, self.objective_value(solution)

class SA:
    def __init__(self, points: list[Point]):
        nodes = len(points)
        self.nodes = nodes
        self.dist = dist = zeros((nodes, nodes), dtype=float64)
        for i in range(nodes):
            for j in range(i, nodes):
                dist[i, j] = dist[j, i] = length(points[i], points[j])
        self.good_neighbors: dict[int, set[int]] = {}
        for i in range(nodes):
            mean = self.dist[i, :].mean()
            self.good_neighbors[i] = set(j for j in range(nodes) if i != j and self.dist[i, j] < 2 * mean)

    def objective_value(self, solution: list[int]):
        nodes = len(solution)
        obj = self.dist[solution[-1], solution[0]]
        for index in range(nodes-1):
            obj += self.dist[solution[index], solution[index+1]]
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
        i = randint(0, self.nodes - 3)
        j = randint(i + 1, self.nodes - 1)
        from1, to1 = solution[i], solution[i+1]
        from2, to2 = solution[j], solution[(j+1) % self.nodes]
        # while True:
        #     if from2 in self.good_neighbors[from1] and to2 in self.good_neighbors[to1]:
        #         break

        new_solution = solution.copy()
        new_solution[i+1:j+1] = solution[i+1:j+1][::-1]
        new_obj = old_obj + dist[from1, from2] + dist[to1, to2] - dist[from1, to1] - dist[from2, to2]
        return new_solution, new_obj

    def three_opt_neighbor(self, solution: ndarray[int32], old_obj: float):
        dist = self.dist
        i = randint(0, self.nodes - 5)
        j = randint(i + 1, self.nodes - 3)
        k = randint(j + 1, self.nodes - 1)
        from1, to1 = solution[i], solution[i+1]
        from2, to2 = solution[j], solution[j+1]
        from3, to3 = solution[k], solution[(k+1) % self.nodes]

        new_solution = solution.copy()
        block1 = solution[i+1:j+1]
        block2 = solution[j+1:k+1]
        new_solution[i+1:k+1] = concatenate((block2[::-1], block1))
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

    def solve(self):
        solution = self.build_greedy_solution()
        obj = self.objective_value(solution)
        best_solution = solution.copy()
        best_obj = obj
        t_max, t_min, steps = self.temperature_heuristic()
        for step in range(steps):
            t = t_max * (t_min / t_max) ** (step / steps)
            new_solution, new_obj = self.three_opt_neighbor(solution, obj)
            accepted = new_obj < best_obj or random() < exp((best_obj - new_obj) / t)
            if accepted:
                solution = new_solution
                obj = new_obj
            if obj < best_obj:
                best_solution = solution
                best_obj = obj
        best_solution, best_obj = self.two_opt_greedy(best_solution, best_obj)
        return best_solution, best_obj