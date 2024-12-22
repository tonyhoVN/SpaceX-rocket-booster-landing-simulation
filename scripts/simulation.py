import matplotlib.pyplot as plt
import numpy as np
import os 
from environment import *
from controller import LQR_Controller, MPC_Controller

#### Simulation setup ####
# initial condition and reference
target_pose = np.array([0.3,0.3,0.3]) # p(x, y, z)
ref_state = np.array([0.3,0.3,0.3,0,0,0,0,0,0,0,0,0]) # p, euler angles, v, omega
initial_pose = np.array([4.0, 4.0, 7.0, 0.3, 0.3, 0.2])

# Create environment
env = Booster(render_mode=True)
state = env.reset(initial_pose=initial_pose, final_target=target_pose)
states = state

# Simulation setup
real_time = 100
steps = int(real_time/0.01) 
sim_time = 100
time_step = sim_time/steps

# create gif
images = []
gif = False

#### Create controller #### 
# Maximum and minimum thrust
u_min = np.array([-4.0, -4.0, 0.0, -2.0, -2.0, -2.0]) 
u_max = np.array([4.0, 4.0, 15.0, 2.0, 2.0, 2.0])

# Kp and Kd gains 
Kp = np.array([1.4,1.4,2.1,10,10,10], dtype=np.float16)
Kd = np.array([2.1,2.1,5.0,10,10,10], dtype=np.float16)
controller = LQR_Controller(Kp, Kd, u_min, u_max)
# controller = MPC_Controller(env.mass, env.I0_inv, u_min, u_max, dT=time_step*10, N=15)
actions = np.zeros((1,6))

#### Main simulation loop ####
num_steps = 0
for i in range(steps):
    num_steps += 1
    
    # calculate desire input
    desire_wrench = controller.update_input(ref_state,state)
    action = controller.update_thrust(desire_wrench)
    
    # perform action
    state, reward, done = env.step(action)
    
    # render 
    render_delay = 0.015 if i < 100 else 0.01
    env.render(render_delay)

    # Save scene image to create gif 
    if gif: 
        image = env.capture_scene()
        if (num_steps%20) == 0:
            images.append(image)

    # add state
    actions = np.vstack((actions,action))
    states = np.vstack((states,state))

    # check done
    if done: break  

# Keep Open3D window run
print("Number of steps: " + str(num_steps))
env.run()
env.close()

# Plot state of booster  
fig = plt.figure()

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.set_aspect('equal')
ax1.plot3D(states[:,0], states[:,1], states[:,2])
ax1.set_title("Booster Trajectory")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(states[:,0], label='x')
ax2.plot(states[:,1], label='y')
ax2.plot(states[:,2], label='z')
ax2.set_title("Position vs Time Step")
ax2.set_xlabel('Time step')
ax2.set_ylabel('Position (m)')
ax2.legend()

ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(states[:,3], label='roll')
ax3.plot(states[:,4], label='pitch')
ax3.plot(states[:,5], label='yaw')
ax3.set_title("Orientation vs Time Step")
ax3.set_xlabel('Time step')
ax3.set_ylabel('Euler angles (rad)')
ax3.legend()

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(states[:,6], label='vx')
ax4.plot(states[:,7], label='vy')
ax4.plot(states[:,8], label='vz')
ax4.set_title("Velocity vs Time Step")
ax4.set_xlabel('Time step')
ax4.set_ylabel('Velocity (m/s)')
ax4.legend()

ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(states[:,9], label='wx')
ax5.plot(states[:,10], label='wy')
ax5.plot(states[:,11], label='wz')
ax5.set_title("Angular Velocity vs Time Step")
ax5.set_xlabel('Time step')
ax5.set_ylabel('Angular Velocity (rad/s)')
ax5.legend()

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(actions)
ax6.set_title("Effort vs Time Step")
ax6.set_xlabel('Time step')
ax6.set_ylabel('Force/Torque (N/Nm)')
ax6.legend(['Fx','Fy','Fz','Mx','My','Mz'])

plt.show()

# Create GIF image 
if gif:
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    image_path = os.path.abspath(os.path.join(folder_path, "images/output.gif"))
    images[0].save(
        image_path,
        save_all=True,
        append_images=images[1:],
        duration=200,  # Duration of each frame in milliseconds
        loop=0  # Loop indefinitely
    )
    print("GIF saved as output.gif")