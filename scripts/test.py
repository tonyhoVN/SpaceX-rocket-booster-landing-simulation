import numpy as np
import cvxpy as cp
import control
from scipy import sparse

# System parameters
m = 1.0  # Mass (kg)
g = 9.8  # Gravitational acceleration (m/s^2)
dt = 0.1  # Time step (s)
N = 15  # Prediction horizon

# State-space matrices
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1 / m]])
C = np.eye(2)
D = np.zeros((2,1))
sys = control.StateSpace(A,B,C,D)
sys_discrete = control.c2d(sys, dt, method='zoh')
A_zoh = sys_discrete.A
B_zoh = sys_discrete.B

# MPC parameters
x_target = np.array([1., 0.])  # Target position
u_min, u_max = -50.0, 50.0  # Force constraints
x0 = np.array([0, 0])  # Initial state [position, velocity]
Q = sparse.diags([100., 10.])
R = np.array([[.1]])

# Optimization variables
x = cp.Variable((2, N + 1))  # State trajectory
u = cp.Variable((1, N))  # Control inputs

# Cost function
cost = 0
constraints = [x[:, 0] == x0]  # Initial state constraint

for k in range(N):
    # Add cost for position and control
    cost += cp.quad_form(x_target - x[:,k], Q) +cp.quad_form(u[:,k], R)
    
    # Dynamics constraints
    constraints += [x[:, k+1] == A_zoh @ x[:, k] + B_zoh @ (u[:, k])]
    
    # Control constraints
    constraints += [u_min <= u[:, k], u[:, k] <= u_max]
    # constraints += [cp.norm(u[:, k], 'inf') <= u_max]

cost += cp.quad_form(x_target - x[:,N], Q)

# Solve the optimization problem
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve(solver=cp.OSQP, warm_start=True)

# Extract the solution
x_opt = x.value
u_opt = u.value

# Display results
print("Optimal state trajectory:", x_opt)
print("Optimal control inputs:", u_opt)

# Plot the results
import matplotlib.pyplot as plt
time = np.arange(N + 1) * dt
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(time, x_opt[0, :], label="Position (x)")
plt.plot(time, x_opt[1, :], label="Velocity (x_dot)")
plt.axhline(x_target[0], color='r', linestyle='--', label="Target")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()

plt.subplot(2, 1, 2)
plt.step(time[:-1], u_opt[0, :], where='post', label="Force (F)")
plt.xlabel("Time (s)")
plt.ylabel("Control Input (F)")
plt.legend()

plt.tight_layout()
plt.show()