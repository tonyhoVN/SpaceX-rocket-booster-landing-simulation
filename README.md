# SpaceX booster landing

This project simulates the return process of a spacecraft booster to its initial launch platform. The booster is designed to execute a controlled descent and align precisely for recovery by a chopstick mechanism. The simulation incorporates Linear Quadratic Regulator (LQR) controllers to achieve optimal control of the booster during its descent and positioning phases. 

<!-- <img src="/images/simulation_env.png" width="480" height="480"> -->
<img src="/images/output.gif">

## Install
Project works in python 3.10

1. Clone the github 

2. Install requirement libraries. 
    ```
    pip install -r requirement.txt
    ```

## Run Simulation
Run file __scripts/simulation.py__ to play simulation.

## Tunning LQR Controller 
In Matlab file __scripts/LQR_tunning.m__, change Q and R parameters to customize behavior of LQR controller. The control result is demonstrated in following figure

<img src="/images/control_result.png">