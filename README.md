# Cart-Pole Model Predictive Control (MPC)

This repository contains a Model Predictive Control (MPC) implementation for a cart-pole system. The goal is to keep the pole balanced by applying forces to the cart using MPC to optimize control inputs over a prediction horizon.

## System Overview

The cart-pole system is a classical control problem, where the objective is to apply forces to a cart such that a pendulum (pole) attached to it remains balanced in the upright position.


## Example Plot

Below is an example plot showing the original and normalized PPG signal over time.

![Normalized PPG Signal](tracking_theta.png)


### State-Space Representation

The linearized dynamics of the system around the upright position are described by the following state-space model:

\[
\dot{x} = Ax + Bu
\]

Where:
- \( x = \begin{bmatrix} \text{position} \\ \text{velocity} \\ \text{angle} \\ \text{angular velocity} \end{bmatrix} \) is the state vector.
- \( u \) is the control input (force applied to the cart).
- \( A \) and \( B \) are the system matrices, which are derived based on the physical parameters of the system.

The state-space matrices are given by:

\[
A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & \frac{m \cdot g}{M} & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & \frac{(M + m) \cdot g}{M \cdot l} & 0
\end{bmatrix}
\quad
B = \begin{bmatrix}
0 \\
\frac{1}{M} \\
0 \\
\frac{1}{M \cdot l}
\end{bmatrix}
\]

### Discretization of the System

The system is discretized using the Zero-Order Hold (ZOH) method. The resulting discrete-time system is:

\[
x_{k+1} = A_d \cdot x_k + B_d \cdot u_k
\]

Where:
- \( A_d \) and \( B_d \) are the discretized matrices computed from the continuous system using the sampling time \( dt \).

### Cost Function

The objective of MPC is to minimize the following cost function over the prediction horizon \( N \):

\[
J = \sum_{k=0}^{N-1} \left( x_k^T Q x_k + u_k^T R u_k \right)
\]

Where:
- \( Q \) and \( R \) are weight matrices that penalize deviations from the desired state and large control inputs, respectively.
- \( x_k \) is the state at time step \( k \).
- \( u_k \) is the control input at time step \( k \).

The optimization problem solved by MPC at each time step is:

\[
\min_{U} \quad J = \sum_{k=0}^{N-1} \left( x_k^T Q x_k + u_k^T R u_k \right)
\]

Where:
- \( U \) is the control input sequence over the prediction horizon \( N \).
- \( x_k \) is the state at each time step.
