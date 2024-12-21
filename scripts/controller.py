from environment import *
from scipy.optimize import lsq_linear


def sgn(x): return (x/abs(x))

class LQR_Controller:
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
    def __init__(self):
        
        # Continuous dynamic model 
        self.A = 0
        self.B = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0.3, 0, 1, 0, 0],
            [-0.3, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])