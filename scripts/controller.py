from environment import *
from scipy.optimize import lsq_linear
import cvxpy as cp
import control
from scipy import sparse
import matplotlib.pyplot as plt


def sgn(x): return (x/abs(x))

class LQR_Controller:
    def __init__(self, Kp , Kd, u_min, u_max):
        """
        LQR controller for booster
        Parameters
        ====
        Kp: proportional gain (6x1)
        Kd: derivative gain (6x1)
        u_min, u_max: thrust constraints (6x1)
        """
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
        self.bounds = (u_min, u_max)
    
    def update_input(self, ref: np.ndarray, state: np.ndarray):
        """
        Calculate desired feedback force of PD controller in body frame
        """
        # state info 
        x,y,z,roll,pitch,yaw,x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot  = state
        R_mat = R.from_euler('ZYX',np.array([yaw,pitch,roll]),degrees=False)

        # Reference
        ref_pos = ref[:3]
        ref_euler = ref[3:6]
        
        # linear error in global
        error_x = ref_pos - np.array([x,y,z])
        error_x_dot = -np.array([x_dot,y_dot,z_dot])

        # linear error in local frame
        error_x_local = (R_mat.as_matrix().T @ error_x.reshape(-1,1)).flatten() 
        error_x_dot_local = (R_mat.as_matrix().T @ error_x_dot.reshape(-1,1)).flatten()

        # rotation error in local frame
        R_error = R_mat.inv()
        q_error = R.as_quat(R_error) #x,y,z,w
        error_eulers_local = sgn(x) * q_error[0:3]
        # error_eulers_local = ref_euler - np.array([roll,pitch,yaw])
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
        Allocate desire wrench to thrust of engines 
        """
        # wrench = desire_wrench[:-1] # drop control on Mz 
        wrench = desire_wrench

        # Solve by pseudo inverse
        # body_thrust = self.A_pinv @ wrench.reshape(-1,1)
        # body_thrust = np.clip(body_thrust.flatten(), self.lower, self.upper)

        # Solve linear least-squares problem 
        result = lsq_linear(
            A = self.A, 
            b = wrench, 
            bounds= self.bounds,
            lsq_solver = 'exact'
        )

        body_thrust = result.x

        return body_thrust.flatten()

class MPC_Controller:
    def __init__(self, m, I0_inv, u_min, u_max, dT=0.01, N=20):
        """
        MPC controller for booster
        
        Parameters
        ====    
        m: mass of booster
        I0_inv: inverse of inertia matrix
        u_min, u_max: thrust constraints
        dT: time step
        N: prediction horizon
        """
        
        # State-space Matrix
        zero = np.zeros((3,3))
        eye = np.eye(3)
        self.A = np.vstack((
            np.hstack((zero,zero,eye,zero,zero)),
            np.hstack((zero,zero,zero,eye,zero)),
            np.hstack((zero,zero,zero,zero,eye)),
            np.hstack((zero,zero,zero,zero,zero)),
            np.hstack((zero,zero,zero,zero,zero)),
        ))
        self.B = np.vstack((
            np.hstack((zero,zero)),
            np.hstack((zero,zero)),
            np.hstack((eye/m,zero)),
            np.hstack((zero,I0_inv)),
            np.hstack((zero,zero)),
        ))
        self.C = np.vstack((
            np.hstack((eye,zero,zero,zero,zero)),
            np.hstack((zero,eye,zero,zero,zero)),
            np.hstack((zero,zero,eye,zero,zero)),
            np.hstack((zero,zero,zero,eye,zero)),
        ))
        self.D = np.zeros((12,6))

        # Control allocation matrix
        self.M = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0.3, 0, 1, 0, 0],
            [-0.3, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        self.M_pinv = np.linalg.pinv(self.M)

        # Weight matrices
        self.Q = sparse.diags([100., 100., 100., 100., 100., 100., 10., 10., 10., 10., 10., 10.])
        self.R = sparse.diags([1., 1., 1., 1., 1., 1.])
        
        # Continuous-time state-space model
        self.sys = control.StateSpace(self.A,self.B,self.C,self.D)
        
        # Discrete-time state-space model
        self.sys_dis = control.c2d(self.sys, dT, method='zoh')
        self.A_dis = self.sys_dis.A
        self.B_dis = self.sys_dis.B
        self.C_dis = self.sys_dis.C
        self.D_dis = self.sys_dis.D
        
        # MPC parameters
        self.dT = dT
        self.N = N

        # Bound of thurst 
        self.u_min = u_min
        self.u_max = u_max
        self.bounds = (u_min, u_max)

    def update_input(self, ref: np.ndarray, state: np.ndarray):
        """
        Solve MPC problem
        """
        # Initial state
        _,_,_,roll,pitch,yaw,x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot  = state
        R_mat = R.from_euler('ZYX',np.array([yaw,pitch,roll]),degrees=False).as_matrix()
        x0 = np.hstack([state, np.array([0,0,-9.8])])

        # Optimization variables
        x = cp.Variable((15, self.N + 1)) # State variables (include g)
        u = cp.Variable((6, self.N)) # Desire control inputs

        # Cost and Constraints
        cost = 0
        constraints = [x[:, 0] == x0]

        # Cost function
        for k in range(self.N):
            # Add cost for position and control
            cost += cp.quad_form(ref - self.C_dis @ x[:,k], self.Q) + cp.quad_form(u[:,k], self.R)

            # Dynamics constraints
            constraints += [x[:, k+1] == self.A_dis @ x[:, k] + self.B_dis @ (u[:, k])]

            # Control constraints
            constraints += [self.u_min <= u[:, k], u[:, k] <= self.u_max]

        cost += cp.quad_form(ref - self.C_dis @ x[:,self.N], self.Q)

        # Solve the optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True)

        # Extract the solution
        desire_input = u.value[:,0] # Take the first control input

        # Convert desire force back to body frame
        desire_input[:3]  = R_mat.T @ desire_input[:3]

        # #
        # # Extract the solution
        # x_opt = x.value[:, :-1]
        # u_opt = u.value

        # time = np.arange(self.N) * self.dT 
        # plt.figure(figsize=(10, 5))

        # plt.subplot(2, 1, 1)
        # plt.plot(time, x_opt[0, :], label="x")
        # plt.plot(time, x_opt[1, :], label="y")
        # plt.plot(time, x_opt[2, :], label="z")
        # plt.legend()

        # plt.subplot(2, 1, 2)
        # plt.step(time, u_opt[0, :], where='post', label="x")
        # plt.step(time, u_opt[1, :], where='post', label="y")
        # plt.step(time, u_opt[2, :], where='post', label="z")
        # plt.legend()
        # plt.show()


        return desire_input.flatten()
        
    
    def update_thrust(self, desire_wrench: np.ndarray):
        """
        Allocate desire wrench to thrust of engines 
        """

        # Solve linear least-squares problem 
        result = lsq_linear(
            A = self.M, 
            b = desire_wrench, 
            bounds= self.bounds,
            lsq_solver = 'exact'
        )

        body_thrust = result.x

        # Solve by pseudo inverse
        # body_thrust = self.M_pinv @ desire_wrench.reshape(-1,1)

        return body_thrust.flatten()