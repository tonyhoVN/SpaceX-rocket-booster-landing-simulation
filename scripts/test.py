import numpy as np
import cvxpy as cp
import control
from scipy import sparse

# System parameters
m = 1.0  # Mass (kg)
g = 9.8  # Gravitational acceleration (m/s^2)
dt = 0.1  # Time step (s)
N = 20  # Prediction horizon

# State-space matrices
A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
B = np.array([[0], [1 / m], [0]])
C = np.array([[1, 0, 0], [0, 1, 0]])
D = np.zeros((2,1))
sys = control.StateSpace(A,B,C,D)
sys_discrete = control.c2d(sys, dt, method='zoh')
A_zoh = sys_discrete.A
B_zoh = sys_discrete.B
C_zoh = sys_discrete.C
D_zoh = sys_discrete.D

# MPC parameters
y_target = np.array([-3., 0.])  # Target position
u_min, u_max = 0.0, 50.0  # Force constraints
x0 = np.array([3.0, 0.0])  # Initial state [position, velocity]
x0 = np.hstack((x0, -9.8))
Q = sparse.diags([100., 20.])
R = np.array([[.1]])

# Optimization variables
x = cp.Variable((3, N + 1))  # State trajectory
u = cp.Variable((1, N))  # Control inputs

# Cost function
cost = 0
constraints = [x[:, 0] == x0]  # Initial state constraint

for k in range(N):
    # Add cost for position and control
    cost += cp.quad_form(y_target - C_zoh @ x[:,k], Q) +cp.quad_form(u[:,k], R)
    
    # Dynamics constraints
    constraints += [x[:, k+1] == A_zoh @ x[:, k] + B_zoh @ (u[:, k])]
    
    # Control constraints
    constraints += [u_min <= u[:, k], u[:, k] <= u_max]
    # constraints += [cp.norm(u[:, k], 'inf') <= u_max]

cost += cp.quad_form(y_target - C_zoh @ x[:,N], Q)

# Solve the optimization problem
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve(solver=cp.OSQP, warm_start=True)

# Extract the solution
x_opt = x.value[:, :-1]
u_opt = u.value

# Display results
print("Optimal state trajectory:", x_opt)
print("Optimal control inputs:", u_opt)

# Plot the results
import matplotlib.pyplot as plt
time = np.arange(N) * dt
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(time, x_opt[0, :], label="Position (x)")
plt.plot(time, x_opt[1, :], label="Velocity (x_dot)")
plt.axhline(y_target[0], color='r', linestyle='--', label="Target")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()

plt.subplot(2, 1, 2)
plt.step(time, u_opt[0, :], where='post', label="Force (F)")
plt.xlabel("Time (s)")
plt.ylabel("Control Input (F)")
plt.legend()

plt.tight_layout()
plt.show()