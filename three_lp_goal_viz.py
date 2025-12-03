# three_lp_goal_viz.py
"""
Helper utilities to add a goal marker to visualize_trajectory playback.
We wrap the pybind visualizer by pre-pending an Open3D mesh for the goal.
"""
import numpy as np


def add_goal_marker(plotter, goal_xy, z=0.0, radius=0.05):
    try:
        import pyvista as pv
    except Exception:
        return None
    center = (float(goal_xy[0]), float(goal_xy[1]), float(z))
    sphere = pv.Sphere(radius=radius, center=center)
    actor = plotter.add_mesh(sphere, color="red")
    return actor


def update_goal_marker(actor, goal_xy, z=0.0):
    if actor is None:
        return
    center = np.array([float(goal_xy[0]), float(goal_xy[1]), float(z)])
    actor.SetPosition(center)  # pyvista actor

