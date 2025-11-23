from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from environment import BoxObstacle, CircleObstacle, PolygonObstacle, EllipseObstacle


def draw_plan(environment, tree, path, filepath, title, stats=None):
    filepath = Path(filepath)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(environment.bounds[0])
    ax.set_ylim(environment.bounds[1])
    for obs in environment.obstacles:
        if isinstance(obs, BoxObstacle):
            span = obs.upper - obs.lower
            ax.add_patch(patches.Rectangle(obs.lower, span[0], span[1], color="#333333", alpha=0.35))
        elif isinstance(obs, CircleObstacle):
            ax.add_patch(patches.Circle(obs.center, obs.radius, color="#333333", alpha=0.35))
        elif isinstance(obs, PolygonObstacle):
            ax.add_patch(patches.Polygon(obs.vertices, color="#333333", alpha=0.35))
        elif isinstance(obs, EllipseObstacle):
            ellipse = patches.Ellipse(obs.center, obs.width * 2, obs.height * 2, 
                                     angle=np.degrees(obs.angle), color="#333333", alpha=0.35)
            ax.add_patch(ellipse)
    for node in tree.nodes:
        if node.parent is None:
            continue
        parent = tree.nodes[node.parent]
        xs = [parent.point[0], node.point[0]]
        ys = [parent.point[1], node.point[1]]
        ax.plot(xs, ys, color="#4f7cac", linewidth=0.6)
    points = np.array([node.point for node in tree.nodes])
    ax.scatter(points[:, 0], points[:, 1], s=4, color="#1f3b4d")
    ax.scatter(environment.start[0], environment.start[1], c="#4caf50", s=80, marker="o")
    ax.scatter(environment.goal[0], environment.goal[1], c="#f44336", s=80, marker="x")
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color="#ff9800", linewidth=2.0)
    ax.set_aspect("equal")
    
    info_text = title
    if stats:
        goal_found = "Yes" if stats.get("goal_found", False) else "No"
        node_count = stats.get("node_count", len(tree.nodes))
        time_val = stats.get("time", None)
        info_text += f"\nGoal: {goal_found}, Nodes: {node_count}"
        if time_val is not None:
            info_text += f", Time: {time_val:.3f}s"
    ax.set_title(info_text, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)

