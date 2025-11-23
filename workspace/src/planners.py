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

    def add(self, point, parent, cost=None):
        if cost is None:
            parent_point = self.nodes[parent].point
            cost = self.nodes[parent].cost + np.linalg.norm(point - parent_point)
        node = TreeNode(point=point, parent=parent, cost=cost)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def update_parent(self, node_index, new_parent, new_cost):
        self.nodes[node_index].parent = new_parent
        self.nodes[node_index].cost = new_cost


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


class RRTStarPlanner(BasePlanner):
    def __init__(self, environment, step_size=0.5, goal_radius=0.5, gamma=30.0, seed=None):
        super().__init__(environment, step_size, goal_radius, seed)
        self.gamma = float(gamma)
        self.dim = len(environment.start)

    def steer(self, from_point, to_point):
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance == 0:
            return from_point
        if distance <= self.step:
            return to_point
        direction /= distance
        return from_point + direction * self.step

    def near_radius(self, n):
        if n <= 1:
            return float("inf")
        return self.gamma * (np.log(n) / n) ** (1.0 / self.dim)

    def near(self, point):
        radius = self.near_radius(len(self.tree.nodes))
        near_nodes = []
        for i, node in enumerate(self.tree.nodes):
            if np.linalg.norm(node.point - point) <= radius:
                near_nodes.append(i)
        return near_nodes

    def choose_parent(self, x_near, x_new):
        x_min = x_near
        c_min = self.tree.nodes[x_near].cost + np.linalg.norm(self.tree.nodes[x_near].point - x_new)
        near_set = self.near(x_new)
        for i in near_set:
            node = self.tree.nodes[i]
            cost = node.cost + np.linalg.norm(node.point - x_new)
            if cost < c_min:
                if not self.env.segment_in_collision(node.point, x_new):
                    x_min = i
                    c_min = cost
        return x_min, c_min

    def rewire(self, x_new_index, x_new_point, new_cost):
        near_set = self.near(x_new_point)
        for i in near_set:
            if i == x_new_index:
                continue
            node = self.tree.nodes[i]
            alt_cost = new_cost + np.linalg.norm(node.point - x_new_point)
            if alt_cost < node.cost:
                if not self.env.segment_in_collision(x_new_point, node.point):
                    self.tree.update_parent(i, x_new_index, alt_cost)

    def plan(self, iterations):
        self.goal_index = None
        for _ in range(iterations):
            x_rand = self.env.sample_free(self.rng)
            x_near_index = self.nearest(x_rand)
            x_near_point = self.tree.nodes[x_near_index].point
            x_new = self.steer(x_near_point, x_rand)
            if self.env.in_collision(x_new):
                continue
            if self.env.segment_in_collision(x_near_point, x_new):
                continue
            x_min_index, c_min = self.choose_parent(x_near_index, x_new)
            x_new_index = self.tree.add(x_new, x_min_index, c_min)
            self.rewire(x_new_index, x_new, c_min)
            if self.env.goal_reached(x_new, self.goal_radius):
                self.goal_index = x_new_index
                break
        return {
            "goal_found": self.goal_index is not None,
            "path": self.path(),
            "node_count": len(self.tree.nodes),
        }

    def nearest(self, point):
        distances = [np.linalg.norm(node.point - point) for node in self.tree.nodes]
        return int(np.argmin(distances))


class RRTStarImprovedPlanner(RRTStarPlanner):
    def cost_local(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def cost_global(self, x):
        return np.linalg.norm(x - self.env.goal)

    def choose_parent(self, x_near, x_new):
        x_min = x_near
        node_near = self.tree.nodes[x_near]
        c_min = (node_near.cost + 
                self.cost_local(node_near.point, x_new) + 
                self.cost_global(node_near.point))
        near_set = self.near(x_new)
        for i in near_set:
            node = self.tree.nodes[i]
            cost = (node.cost + 
                   self.cost_local(node.point, x_new) + 
                   self.cost_global(node.point))
            if cost < c_min:
                if not self.env.segment_in_collision(node.point, x_new):
                    x_min = i
                    c_min = cost
        node_min = self.tree.nodes[x_min]
        actual_cost = node_min.cost + self.cost_local(node_min.point, x_new)
        return x_min, actual_cost

    def rewire(self, x_new_index, x_new_point, new_cost):
        near_set = self.near(x_new_point)
        for i in near_set:
            if i == x_new_index:
                continue
            node = self.tree.nodes[i]
            alt_cost = new_cost + self.cost_local(x_new_point, node.point)
            if alt_cost < node.cost:
                if not self.env.segment_in_collision(x_new_point, node.point):
                    self.tree.update_parent(i, x_new_index, alt_cost)

