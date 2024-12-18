clear;clc

% Model Discrete System
m = 1.0;
A = [0 1; 0 0];
B = [0 1/m]';
C = eye(2);
D = [0 0]';
Ts = 0.01;
sys_c = ss(A,B,C,D);
sys_d = c2d(sys_c, Ts, 'zoh');
A_d = sys_d.A;
B_d = sys_d.B;
C_d = sys_d.C;
D_d = sys_d.D;

% Choose Q and R
Q = [100 0; 0 20];   
R = 10;       

% Compute the LQR gain K
K = dlqr(A_d, B_d, Q, R);

% Feedback model and step response
sys1 = ss(A_d-B_d*K,B_d,C_d,D_d,Ts);
step(sys1)