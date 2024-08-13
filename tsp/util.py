import math
from collections import namedtuple
from turtle import Turtle, Screen, done, screensize

Point = namedtuple("Point", ["x", "y"])

def length(a: Point, b: Point) -> float:
    dx = (a.x - b.x)
    dy = (a.y - b.y)
    return math.sqrt(dx * dx + dy * dy)

def length2(a: Point, b: Point) -> float:
    dx = (a.x - b.x)
    dy = (a.y - b.y)
    return dx * dx + dy * dy

def visualize_solution(points: list[Point], solution: list[int]):
    coordinates: list[tuple[float, float]] = [(points[a].x * 0.1, points[a].y * 0.1) for a in solution]
    t = Turtle()
    t.shape("circle")
    t.shapesize(0.1)
    t.up()
    t.goto(coordinates[0])
    t.stamp()
    t.down()
    for point in coordinates[1:]:
        t.goto(point)
        t.stamp()
    t.goto(coordinates[0])
    done()
