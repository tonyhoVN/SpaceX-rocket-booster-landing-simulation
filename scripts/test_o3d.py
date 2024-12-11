import open3d as o3d
import numpy as np
import time
from utils import *

# Load booster
booster =  o3d.io.read_triangle_mesh("./models/booster_model.STL")
booster.compute_vertex_normals()

# Transform to center of mesh 
transform_0 = np.eye(4)
transform_0[0,3] = -0.1
transform_0[1,3] = -0.1
transform_0[2,3] = -0.3
booster.transform(transform_0)

# Load chopstick 
chopstick = o3d.io.read_triangle_mesh("./models/chopstick_model.STL")
chopstick.compute_vertex_normals()

# Create the origin marker
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
body = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
# body.translate([0, 0, 0.3])

arrow = draw_force(np.array([0]))

# Combine objects into a scene
scene_objects = [booster, chopstick, origin, body, arrow]

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window(width=700, height=700)

for obj in scene_objects:
    vis.add_geometry(obj)

# # Set the viewing angle
# view_control = vis.get_view_control()
# view_control.set_front([0.5, 1.0, 1.0])  # Set the view front
# view_control.set_lookat([0, 0, 1.0])         # Set the target point
# view_control.set_up([0, 0, 1])             # Set the up direction
# view_control.set_zoom(0.8)                 # Adjust zoom level

# Simulation parameters
velocity = 0.001  # m/s
distance = 1.0  # m
time_steps = int(distance / velocity)
step_time = 0.01 # seconds per step

# Simulate upward movement
for t in range(time_steps + 1):
    # Update the translation in the transformation matrix
    transform = np.eye(4)
    transform[2, 3] = velocity * t# Z-axis movement

    # Apply the transformation to the booster
    booster.transform(transform)
    body.transform(transform)

    # Update the visualization
    vis.update_geometry(booster)
    vis.update_geometry(body)
    vis.poll_events()
    vis.update_renderer()

    # Pause for the next frame
    time.sleep(step_time)

    # Convert back to origin
    inv_transform = np.linalg.inv(transform)
    booster.transform(np.linalg.inv(transform))
    body.transform(np.linalg.inv(transform))

# Close the visualization window after simulation
vis.destroy_window()