import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from environment import PlaneEnvironment, BoxObstacle, CircleObstacle, PolygonObstacle, EllipseObstacle
from planners import RandomTreePlanner, HeuristicTreePlanner, RRTPlanner
from visualizer import draw_plan


def build_environment():
    bounds = np.array([[0.0, 20.0], [0.0, 15.0]])
    start = np.array([1.5, 1.5])
    goal = np.array([18.0, 13.0])
    
    def regular_polygon(center, radius, n_sides, rotation=0):
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + rotation
        return np.array([[center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)] for a in angles])
    
    obstacles = [
        BoxObstacle([3.0, 1.5], [6.5, 4.0]),
        CircleObstacle([9.0, 3.0], 1.8),
        PolygonObstacle(regular_polygon([12.0, 2.5], 1.5, 3, np.pi / 6)),
        EllipseObstacle([15.0, 3.5], 1.8, 1.2, np.pi / 4),
        PolygonObstacle(regular_polygon([5.0, 7.5], 1.3, 5)),
        CircleObstacle([9.5, 7.0], 1.5),
        PolygonObstacle(regular_polygon([13.0, 7.5], 1.4, 6, np.pi / 6)),
        BoxObstacle([16.0, 6.0], [18.5, 8.5]),
        PolygonObstacle(regular_polygon([3.5, 10.5], 1.2, 4, np.pi / 4)),
        EllipseObstacle([7.5, 11.0], 1.6, 1.0, -np.pi / 3),
        CircleObstacle([11.0, 10.5], 1.3),
        PolygonObstacle(regular_polygon([14.5, 11.5], 1.5, 7)),
    ]
    return PlaneEnvironment(bounds, start, goal, obstacles, step=0.35)


def run(iterations=800):
    env = build_environment()
    planners = [
        ("part1_random", RandomTreePlanner(env, step_size=0.5, goal_radius=0.6, seed=4)),
        ("part1_heuristic", HeuristicTreePlanner(env, step_size=0.45, goal_radius=0.6, seed=4)),
        ("part1_rrt", RRTPlanner(env, step_size=0.5, goal_radius=0.6, seed=4)),
    ]
    outputs = {}
    for name, planner in planners:
        stats = planner.plan(iterations)
        outputs[name] = stats
        path = planner.path()
        out_file = ROOT / "output" / f"{name}.png"
        title = name.replace("_", " ").title()
        draw_plan(env, planner.tree, path, out_file, title)
    return outputs


if __name__ == "__main__":
    summary = run()
    for key, value in summary.items():
        print(key, value)

