import open3d as o3d
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from scipy.optimize import lsq_linear
from PIL import Image
import os

def log_colored_message(message, color="GREEN"):
    """
    Logs a message in the specified color.

    Args:
        message (str): The message to log.
        color (str): The color of the message. Options are 'GREEN', 'YELLOW', 'RED'.
    """
    # Define ANSI color codes
    COLORS = {
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "RESET": "\033[0m"  # Reset to default color
    }

    # Get the color code
    color_code = COLORS.get(color.upper(), COLORS["RESET"])

    # Print the colored message
    print(f"{color_code}{message}{COLORS['RESET']}")

def validate_array_size(arr):
    # Check if the input is a numpy array
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    
    # Check if the array size is 6
    if arr.size != 6:
        raise ValueError(f"Array must have exactly 6 elements, but got size {arr.size}.")

def get_rpy(rotation_matrix, degree=False):
    # Create a Rotation object from the matrix
    rotation = R.from_matrix(rotation_matrix)

    # Extract Euler angles ZYX (yaw, pitch, roll) of instrinsic rotation
    yaw, pitch, roll = rotation.as_euler('ZYX', degrees=degree)  # Set degrees=False for radians
    return (roll,pitch,yaw)

def cross_product_column_vectors(vec1, vec2):
    """
    Input is column vectors -> Output column vector 
    """
    # Ensure inputs are numpy arrays
    vec1 = np.array(vec1).reshape(-1)
    vec2 = np.array(vec2).reshape(-1)
    
    # Validate size of vectors
    if vec1.size != 3 or vec2.size != 3:
        raise ValueError("Both input vectors must be 3-dimensional.")

    # Compute the cross product
    cross_prod = np.cross(vec1, vec2)
    return np.array([cross_prod]).transpose()

def make_TF(x,y,z,R):
    T = np.eye(4)  # Initialize as identity matrix
    T[:3, :3] = R  # Assign the rotation matrix
    T[:3, 3] = [x,y,z]   # Assign the translation vector
    return T

def w_cross(w):
    w = w.reshape(-1)
    w_cross = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])
    return w_cross

def draw_force(force: np.array, point=np.ones(3), R=np.ones(3)):
    """
    Draw force arrow
    """
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01, 
        cone_radius=0.015, 
        cylinder_height=0.05, 
        cone_height=0.005, 
        resolution=10, 
        cylinder_split=4, 
        cone_split=1
    )
    return arrow

def create_scene(initial_transform = np.eye(4)):
    # Load booster
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    booster = o3d.io.read_triangle_mesh(os.path.abspath(os.path.join(dir, "models/booster_model.STL")))
    booster.compute_vertex_normals()
    # Transform to center of mesh 
    transform_0 = np.eye(4)
    transform_0[0,3] = -0.15
    transform_0[1,3] = -0.15
    transform_0[2,3] = -0.3
    booster.transform(transform_0)
    # Translate to position
    booster.transform(initial_transform)
    # booster.paint_uniform_color([1, 0.706, 0])
    
    # Load chopstick 
    chopstick = o3d.io.read_triangle_mesh(os.path.abspath(os.path.join(dir, "models/chopstick_model.STL")))
    chopstick.compute_vertex_normals()
    chopstick.paint_uniform_color([0.0, 0.7, 0.7])

    # Create the origin marker
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    body = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    body.transform(initial_transform)

    # Create Point cloud to visualize trajectory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(initial_transform[:3,3].reshape(1,-1))

    # Combine objects into a scene
    scene_objects = [booster, body, pcd, chopstick , origin]

    # create visualization 
    camera_parameters = o3d.io.read_pinhole_camera_parameters(os.path.abspath(os.path.join(dir, "scripts/ScreenCamera.json")))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera_parameters.intrinsic.width, 
                      height=camera_parameters.intrinsic.height)
    
    # Add object
    for obj in scene_objects:
        vis.add_geometry(obj)

    # Set the camera view parameters
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera_parameters,  allow_arbitrary=True)

    return vis, scene_objects, camera_parameters

def reset_mesh(mesh, T):
    """
    Reset mesh to original 
    """
    mesh.transform(np.linalg.inv(T))

class Booster():

    def __init__(self, render_mode=True) -> None:
        # State space
        self.state = None # x,y,z,yaw,pitch,roll,x_dot,y_dot,z_dot,yaw_dot,pitch_dot,roll_dot 

        # Property 
        self.mass = 1 # mass 
        self.I0 = np.eye(3) # tensor
        self.I0_inv = np.eye(3)

        self.height = 0.6 # height
        self.radius = 0.1 # radius
        self.mg = self.mass*np.array([[0,0,-9.8]]).transpose() # gravity 
        self.R = np.eye(3, dtype=np.float16) # rotation matrix
        self.T = np.eye(4, dtype=np.float16)

        # Simulation and visualization parameters
        self.DT = 0.01
        self.render_mode = render_mode
        self.vis = None
        self.scene_objects = None
        self.camera_parameters = None
        self.pcd = None # pointcloud to visualize trajectory 

        # Dynamic parameters
        self.p = np.zeros((3,1), dtype=np.float16)
        self.L = np.zeros((3,1), dtype=np.float16)
        self.w = np.zeros((3,1), dtype=np.float16)

        # Limit for termination
        self.threshold_linear = 0.1
        self.threshold_angular = np.deg2rad(45)
        

    def reset(self, initial_pose: np.ndarray, final_target: np.ndarray):
        """
        Reset the environment to the initial state.
        """
        # Target and intial
        self.target = final_target
        self.initial_pose = initial_pose

        # initial state
        x,y,z = initial_pose[:3]
        roll,pitch,yaw = initial_pose[3:]
        x_dot,y_dot,z_dot,yaw_dot,pitch_dot,roll_dot = [0]*6
        
        self.state = np.array([x,y,z,
                               roll,pitch,yaw,
                               x_dot,y_dot,z_dot,
                               roll_dot,pitch_dot,yaw_dot], dtype=np.float16)
        self.R = R.from_euler('ZYX',np.array([yaw,pitch,roll]),degrees=False).as_matrix()

        # transformation 
        self.T = make_TF(x,y,z,self.R)

        # create visualization scene
        if self.render_mode:
            self.vis, self.scene_objects, self.camera_parameters = create_scene(self.T)
            self.pcd = self.scene_objects[2]
    
        return self.state

    def step(self, action: np.ndarray):
        """
        Update step for forward dynamics
        """
        # state 
        x,y,z,roll,pitch,yaw,x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot  = self.state
        X = np.array([x,y,z]).reshape(-1,1)
        w_b =np.array([roll_dot,pitch_dot,yaw_dot]).reshape(-1,1)
        
        # F/T input
        # validate_array_size(action)
        F_input = action[:3].reshape(-1,1)
        M_input = action[3:].reshape(-1,1)

        # F/T in body frame
        F_b = F_input
        r = np.array([0,0,-self.height/2]).reshape(-1,1) # vector from C.O.M to force input
        M_b = M_input + cross_product_column_vectors(r, F_b)
        # M_b = M_input
        # M_b = cross_product_column_vectors(r, F_b)
        # M_b = np.zeros((3,1))

        # Global input
        F_global = np.matmul(self.R, F_b) + self.mg
        M_global = np.matmul(self.R, M_b)

        # Translation w.r.t fixed frame
        dp = F_global*self.DT
        self.p = self.p + dp
        X_dot = self.p/self.mass
        X = X + X_dot*self.DT

        # Rotation w.r.t fixed frame        
        dL = M_global*self.DT
        self.L = self.L + dL
        w = self.R@self.I0_inv@(self.R.T@self.L)
        R_dot = np.dot(w_cross(w), self.R)
        self.R = self.R + R_dot*self.DT
        
        # self.L = (self.R @ self.I0_inv @ self.R.T) @ w

        # Euler angle
        eulers = np.array(get_rpy(self.R)).reshape(-1,1)

        # Stack the state
        self.state = np.row_stack([X, eulers, X_dot, self.R.T@w]).flatten() # convert back to 1D array
        
        # reward
        reward = self.compute_reward(action)

        # check done conditions 
        done = self.check_done()

        return self.state, reward, done

    def compute_reward(self, action: np.ndarray):
        # State
        x,y,z,roll,pitch,yaw,x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot  = self.state

        # Position error
        position_error = np.linalg.norm(np.array([x,y,z]) - self.target)
        position_reward = -position_error  # Penalize distance

        # Rotation error
        orientation_error = abs(roll) + abs(pitch) + abs(yaw)
        orientation_reward = -orientation_error

        # Velocity penalties 
        vel_linear = -np.linalg.norm(np.array([x_dot,y_dot,z_dot]))
        vel_rotation = -np.linalg.norm(np.array([roll_dot,pitch_dot,yaw_dot]))

        # reward symmetric 
        x_waypoint_target = (self.initial_pose[0]/self.initial_pose[2])*z 
        y_waypoint_target = (self.initial_pose[1]/self.initial_pose[2])*z
        waypoint_reward = -np.linalg.norm(np.array([x,y]) - np.array([x_waypoint_target, y_waypoint_target]))

        # Effort control reward 
        effort_penalty = -np.linalg.norm(action)

        # Goal achievement reward 
        if (position_error <= self.threshold_linear) and (orientation_error <= 0.1):
            goal_reward = 100
        else:
            goal_reward = 0

        # Total reward 
        reward = 1.0 * position_reward + \
                 1.0 * orientation_reward + \
                 1.0 * waypoint_reward + \
                 1.0 * goal_reward + \
                 0.5 * vel_linear + \
                 0.5 * vel_rotation + \
                 0.1 * effort_penalty

        return reward 

    def check_done(self):
        """
        Check finish mission
        """
        X = self.state[:3]
        roll,pitch,yaw = self.state[3:6]
        
        if np.linalg.norm(X-self.target) <= self.threshold_linear:
            log_colored_message("MISSION SUCCESS", "GREEN")
            return True
        
        if abs(roll) >= self.threshold_angular or \
           abs(pitch) >= self.threshold_angular or \
           abs(yaw) >= self.threshold_angular:
            log_colored_message("ROTATION FAIL", "YELLOW")
            return True
        
        if X[0] < -1 or \
           X[1] < -1 or \
           X[2] < -1 or \
            X[0] > self.initial_pose[0] + 0.5 or \
            X[1] > self.initial_pose[1] + 0.5 or \
            X[2] > self.initial_pose[2] + 0.5:
            log_colored_message("LOCATION FAIL", "RED")
            return True

        return False


    def render(self, time_delay=0.01):
        """
        Render the scene 
        """
        # Remove object 
        reset_mesh(self.scene_objects[0], self.T)
        reset_mesh(self.scene_objects[1], self.T)

        # update transformation of body 
        self.T = make_TF(self.state[0],self.state[1],self.state[2],self.R)

        # Update the visualization
        self.scene_objects[0].transform(self.T)
        self.scene_objects[1].transform(self.T)
        self.pcd.points.extend([self.state[:3]])
        self.vis.update_geometry(self.scene_objects[0])
        self.vis.update_geometry(self.scene_objects[1])
        self.vis.update_geometry(self.pcd)

        # Set the camera parameters
        
        # view_control = self.vis.get_view_control()
        # view_control.convert_from_pinhole_camera_parameters(self.camera_parameters)

        # update view window
        self.vis.poll_events()
        self.vis.update_renderer()

        # sleep to display
        time.sleep(time_delay)

    def run(self):
        """
        Keep window running
        """
        self.vis.run()

    def close(self):
        """
        close env
        """
        self.vis.close()

    def capture_scene(self):
        """
        Capture scene image to create gif
        """
        float_buffer = self.vis.capture_screen_float_buffer(do_render=False)
        scene_image = (np.asarray(float_buffer) * 255).astype(np.uint8)
        scene_image = Image.fromarray(scene_image)
        return scene_image

def sgn(x): return (x/abs(x))

class Controller():
    def __init__(self, Kp , Kd):
        self.Kp = np.array(Kp, dtype=np.float16)
        self.Kd = np.array(Kd, dtype=np.float16)
        
        # Control allocation matrix
        self.A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0.3, 0, 1, 0, 0],
            [-0.3, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        self.A_pinv = np.linalg.pinv(self.A)

        # Bound of thurst 
        self.lower = np.array([-2.0, -2.0, 0.0, -1.0, -1.0, -1.0])
        self.upper = np.array([2.0, 2.0, 15.0, 1.0, 1.0, 1.0])
        self.bounds = (self.lower, self.upper)
    
    def update_PD(self, ref: np.ndarray, state: np.ndarray):
        """
        Calculate desired feedback force of PD controller in body frame
        """
        # state info 
        x,y,z,roll,pitch,yaw,x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot  = state
        R_mat = R.from_euler('ZYX',np.array([yaw,pitch,roll]),degrees=False)
        
        # linear error in global
        error_x = ref - np.array([x,y,z])
        error_x_dot = -np.array([x_dot,y_dot,z_dot])

        # linear error in local frame
        error_x_local = (R_mat.as_matrix().T @ error_x.reshape(-1,1)).flatten() 
        error_x_dot_local = (R_mat.as_matrix().T @ error_x_dot.reshape(-1,1)).flatten()

        # rotation error in local frame
        R_error = R_mat.inv()
        q_error = R.as_quat(R_error) #x,y,z,w
        error_eulers_local = sgn(x) * q_error[0:3]
        # error_eulers_local = -np.array([roll,pitch,yaw])
        error_eulers_dot_local = -np.array([roll_dot,pitch_dot,yaw_dot])
        
        # Calculate desired force
        error_local = np.hstack([error_x_local, error_eulers_local]).flatten()
        error_dot_local = np.hstack([error_x_dot_local, error_eulers_dot_local])
        desire_input = self.Kp*error_local + self.Kd*error_dot_local

        # Add g compensation for force component 
        mg = np.array([0,0,-9.8]).reshape(-1,1)
        desire_input[:3] = desire_input[:3] - (R_mat.as_matrix().T @ mg).flatten()

        return desire_input

    def update_thrust(self, desire_wrench: np.ndarray):
        """
        Convert desire wrench to thrust of engines 
        """
        # wrench = desire_wrench[:-1] # drop control on Mz 
        wrench = desire_wrench

        # Solve by pseudo inverse
        body_thrust = self.A_pinv @ wrench.reshape(-1,1)
        body_thrust = np.clip(body_thrust.flatten(), self.lower, self.upper)

        # Solve linear least-squares problem 
        # result = lsq_linear(
        #     A = self.A, 
        #     b = wrench, 
        #     bounds= self.bounds,
        #     lsq_solver = 'exact'
        # )

        # body_thrust = result.x

        # apend Mz 
        # body_thrust = np.hstack((body_thrust,desire_wrench[-1]))

        return body_thrust.flatten()
