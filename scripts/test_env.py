import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *

# Function to plot a frame represented by a transformation matrix
def plot_frame(ax, T, frame_name="Frame", color='r', length=1.0):
    # Extract origin and rotation matrix
    origin = T[:3, 3]
    R = T[:3, :3]

    # Define axes
    x_axis = origin + R[:, 0] * length  # X-axis
    y_axis = origin + R[:, 1] * length  # Y-axis
    z_axis = origin + R[:, 2] * length  # Z-axis

    # Plot origin
    ax.scatter(*origin, color=color, label=f"{frame_name} Origin")

    # Plot axes
    ax.quiver(*origin, *(x_axis - origin), color='r', label=f"{frame_name} X", arrow_length_ratio=0.1)
    ax.quiver(*origin, *(y_axis - origin), color='g', label=f"{frame_name} Y", arrow_length_ratio=0.1)
    ax.quiver(*origin, *(z_axis - origin), color='b', label=f"{frame_name} Z", arrow_length_ratio=0.1)

# Create controller 
Kp = np.array([1.4,1.4,2.1,10,10,10], dtype=np.float16)
Kd = np.array([2.0,2.0,5.0,10,10,10], dtype=np.float16)
controller = Controller(Kp, Kd)
actions = np.zeros((1,6))

# initial condition and reference
ref = np.array([0.3,0.3,0.3])
initial_pose = np.array([2.0, 2.0, 7.0, 0.3, 0.3, 0.0])

# Create environment
env = Booster(render_mode=True)
state = env.reset(initial_pose=initial_pose, final_target=ref)
states = state

# Simulation setup
real_time = 100
steps = int(real_time/0.01) 
sim_time = 100
time_step = sim_time/steps

# create figure

# Main simulation loop
num_steps = 0
for i in range(steps):
    
    # calculate require wrench from controller 
    desire_wrench = controller.update_PD(ref,state)
    action = controller.update_thrust(desire_wrench)
    
    # perform action
    state, reward, done = env.step(action)
    
    # render 
    render_delay = 0.01 if i < 100 else 0.01
    env.render(render_delay)

    num_steps += 1
    # # Transform the axes vectors
    # T = make_TF(state[0], state[1], state[2], env.R)
    # T1 = make_TF(state[0], state[1], state[2],np.eye(3))

    # # Clear previous plot
    # ax.cla()
    # # Plot the updated frame
    # plot_frame(ax, np.eye(4), frame_name="Origin", color='k')  # Origin frame
    # plot_frame(ax, T, frame_name=f"Step", color='r')  # Updated frame
    # plot_frame(ax, T1, frame_name=f"Step", color='b')

    # # Set axis limits
    # ax.set_xlim([-1, 5])
    # ax.set_ylim([-1, 5])
    # ax.set_zlim([0, 5])
    # plt.pause(0.01)

    # add state
    actions = np.vstack((actions,action))
    states = np.vstack((states,state))

    # check done
    if done: break  

print("Number of steps: " + str(num_steps))
env.vis.run()
env.close()

# Plot state of booster  
fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_aspect('equal')
ax1.plot3D(states[:,0], states[:,1], states[:,2])
ax1.set_title("Booster Trajectory")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(states[:,0], label='x')
ax2.plot(states[:,1], label='y')
ax2.plot(states[:,2], label='z')
ax2.set_title("Position vs Time Step")
ax2.set_xlabel('Time step')
ax2.set_ylabel('Position (m)')
ax2.legend()

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(states[:,3], label='roll')
ax3.plot(states[:,4], label='pitch')
ax3.plot(states[:,5], label='yaw')
ax3.set_title("Orientation vs Time Step")
ax3.set_xlabel('Time step')
ax3.set_ylabel('Euler angles (rad)')
ax3.legend()

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(actions)
ax4.set_title("Effort vs Time Step")
ax4.set_xlabel('Time step')
ax4.set_ylabel('Force/Torque (N/Nm)')
ax4.legend(['Fx','Fy','Fz','Mx','My','Mz'])

plt.show()