from dataclasses import dataclass
import numpy as np


@dataclass
class TreeNode:
    point: np.ndarray
    parent: int | None
    cost: float


class SearchTree:
    def __init__(self, root):
        self.nodes = [TreeNode(point=np.array(root, dtype=float), parent=None, cost=0.0)]

    def add(self, point, parent):
        parent_point = self.nodes[parent].point
        cost = self.nodes[parent].cost + np.linalg.norm(point - parent_point)
        node = TreeNode(point=point, parent=parent, cost=cost)
        self.nodes.append(node)
        return len(self.nodes) - 1


class BasePlanner:
    def __init__(self, environment, step_size=0.5, goal_radius=0.5, seed=None):
        self.env = environment
        self.step = float(step_size)
        self.goal_radius = float(goal_radius)
        self.tree = SearchTree(environment.start)
        self.goal_index = None
        self.rng = np.random.default_rng(seed)

    def choose_connection(self):
        return self.select_node(), self.select_target()

    def select_node(self):
        raise NotImplementedError

    def select_target(self):
        return self.env.sample_free(self.rng)

    def extend(self, node_index, target):
        origin = self.tree.nodes[node_index].point
        direction = target - origin
        distance = np.linalg.norm(direction)
        if distance == 0:
            return None
        direction /= distance
        remaining = distance
        current = origin
        parent = node_index
        last = None
        while remaining > 1e-9:
            step = min(self.step, remaining)
            candidate = current + direction * step
            if self.env.segment_in_collision(current, candidate):
                break
            parent = self.tree.add(candidate, parent)
            current = candidate
            remaining -= step
            last = parent
        return last

    def plan(self, iterations):
        self.goal_index = None
        for _ in range(iterations):
            node_index, target = self.choose_connection()
            new_index = self.extend(node_index, target)
            if new_index is None:
                continue
            point = self.tree.nodes[new_index].point
            if self.env.goal_reached(point, self.goal_radius):
                self.goal_index = new_index
                break
        return {
            "goal_found": self.goal_index is not None,
            "path": self.path(),
            "node_count": len(self.tree.nodes),
        }

    def path(self):
        if self.goal_index is None:
            return None
        chain = []
        index = self.goal_index
        while index is not None:
            chain.append(self.tree.nodes[index].point)
            index = self.tree.nodes[index].parent
        chain.reverse()
        return np.stack(chain)


class RandomTreePlanner(BasePlanner):
    def select_node(self):
        return int(self.rng.integers(0, len(self.tree.nodes)))


class HeuristicTreePlanner(BasePlanner):
    def select_node(self):
        goal = self.env.goal
        scores = [np.linalg.norm(node.point - goal) for node in self.tree.nodes]
        return int(np.argmin(scores))


class RRTPlanner(BasePlanner):
    def choose_connection(self):
        target = self.env.sample_free(self.rng)
        index = self.nearest(target)
        return index, target

    def nearest(self, point):
        distances = [np.linalg.norm(node.point - point) for node in self.tree.nodes]
        return int(np.argmin(distances))

