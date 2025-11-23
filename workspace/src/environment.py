from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


class Obstacle(ABC):
    @abstractmethod
    def contains(self, point: np.ndarray) -> bool:
        pass

    def intersects(self, a: np.ndarray, b: np.ndarray, step: float) -> bool:
        direction = b - a
        length = np.linalg.norm(direction)
        if length == 0:
            return self.contains(a)
        count = max(1, int(np.ceil(length / step)))
        for i in range(count + 1):
            p = a + direction * (i / count)
            if self.contains(p):
                return True
        return False


@dataclass
class BoxObstacle(Obstacle):
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        self.lower = np.array(self.lower, dtype=float)
        self.upper = np.array(self.upper, dtype=float)

    def contains(self, point: np.ndarray) -> bool:
        return np.all(point >= self.lower) and np.all(point <= self.upper)


@dataclass
class CircleObstacle(Obstacle):
    center: np.ndarray
    radius: float

    def __post_init__(self):
        self.center = np.array(self.center, dtype=float)
        self.radius = float(self.radius)

    def contains(self, point: np.ndarray) -> bool:
        return np.linalg.norm(point - self.center) <= self.radius


@dataclass
class PolygonObstacle(Obstacle):
    vertices: np.ndarray

    def __post_init__(self):
        self.vertices = np.array(self.vertices, dtype=float)

    def contains(self, point: np.ndarray) -> bool:
        n = len(self.vertices)
        inside = False
        j = n - 1
        for i in range(n):
            vi = self.vertices[i]
            vj = self.vertices[j]
            if ((vi[1] > point[1]) != (vj[1] > point[1])) and \
               (point[0] < (vj[0] - vi[0]) * (point[1] - vi[1]) / (vj[1] - vi[1]) + vi[0]):
                inside = not inside
            j = i
        return inside


@dataclass
class EllipseObstacle(Obstacle):
    center: np.ndarray
    width: float
    height: float
    angle: float = 0.0

    def __post_init__(self):
        self.center = np.array(self.center, dtype=float)
        self.width = float(self.width)
        self.height = float(self.height)
        self.angle = float(self.angle)
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        self.rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        self.inv_rotation = self.rotation.T

    def contains(self, point: np.ndarray) -> bool:
        translated = point - self.center
        rotated = self.inv_rotation @ translated
        return (rotated[0] / self.width) ** 2 + (rotated[1] / self.height) ** 2 <= 1.0


class PlaneEnvironment:
    def __init__(self, bounds, start, goal, obstacles=None, step=0.5):
        self.bounds = np.array(bounds, dtype=float)
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = obstacles or []
        self.step = float(step)

    def in_bounds(self, point):
        return np.all(point >= self.bounds[:, 0]) and np.all(point <= self.bounds[:, 1])

    def in_collision(self, point):
        if not self.in_bounds(point):
            return True
        for obs in self.obstacles:
            if obs.contains(point):
                return True
        return False

    def segment_in_collision(self, a, b):
        direction = b - a
        length = np.linalg.norm(direction)
        if length == 0:
            return self.in_collision(a)
        count = max(1, int(np.ceil(length / self.step)))
        for i in range(count + 1):
            p = a + direction * (i / count)
            if self.in_collision(p):
                return True
        return False

    def sample_free(self, rng):
        for _ in range(1000):
            point = rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            if not self.in_collision(point):
                return point
        raise RuntimeError("failed to sample free configuration")

    def goal_reached(self, point, radius):
        return np.linalg.norm(point - self.goal) <= radius

