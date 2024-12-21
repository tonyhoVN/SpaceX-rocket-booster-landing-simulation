import matplotlib.pyplot as plt
import numpy as np
import os 
from environment import *
from controller import LQR_Controller

# Create controller 
Kp = np.array([1.4,1.4,2.1,10,10,10], dtype=np.float16)
Kd = np.array([2.1,2.1,5.0,10,10,10], dtype=np.float16)
controller = LQR_Controller(Kp, Kd)
actions = np.zeros((1,6))

# initial condition and reference
ref = np.array([0.3,0.3,0.3])
initial_pose = np.array([2.0, 2.0, 7.0, 0.3, 0.3, 0.2])

# Create environment
env = Booster(render_mode=True)
state = env.reset(initial_pose=initial_pose, final_target=ref)
states = state

# Simulation setup
real_time = 100
steps = int(real_time/0.01) 
sim_time = 100
time_step = sim_time/steps

# create gif
images = []
gif = False

# Main simulation loop
num_steps = 0
for i in range(steps):
    num_steps += 1
    
    # calculate require wrench from controller 
    desire_wrench = controller.update_PD(ref,state)
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