import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Cart-pole system parameters
m = 0.2     # Mass of the pendulum (kg)
M = 1.0     # Mass of the cart (kg)
l = 0.5    # Length of the pendulum (m)
g = 9.81   # Gravity (m/s^2)
dt = 0.05   # Time step (s)

# Linearized cart-pole dynamics (around the upright position)
A = np.array([
    [0, 1, 0, 0],
    [0, 0, m * g / M, 0],
    [0, 0, 0, 1],
    [0, 0, (M + m) * g / (M * l), 0]
])

B = np.array([[0], [1 / M], [0], [1 / (M * l)]])

# Discretize the system using Zero-Order Hold (ZOH) method
Ad, Bd, _, _, _ = scipy.signal.cont2discrete((A, B, np.eye(4), 0), dt)
# print(f"Ad shape: {Ad.shape}, Bd shape: {Bd.shape}")  (4,4) (4,1)


# Cost matrices
Q = np.diag([10, 1, 50, 1])  # Penalize position, velocity, angle, and angular velocity    10 position 50 angles
R = np.array([[0.1]])        # Penalize control input

# MPC parameters
N = 20  # Prediction horizon

def cartpole_dynamics(x, u):
    u = float(u)  # Ensure u is a scalar
    
    x = x.reshape(4, 1)  # Ensure x is (4, 1)
    
    print(f"x shape: {x.shape}, u value: {u}")  # Updated to print the value of u

    return Ad @ x + Bd * u

def mpc_cost(U, x0):
    U = U.reshape(N, 1)  # Ensure U is a column vector of shape (N, 1)
    x = x0.reshape(-1, 1)  # Ensure x0 is a (4, 1) column vector
    cost = 0
    
    
    for i in range(N):
        u = U[i].reshape(1, 1)  # Ensure u is a (1, 1) column vector
        x = cartpole_dynamics(x, u)  # Output should be (4, 1)
        
        #print(f"x shape: {x.shape}, u shape: {u.shape}")
        
        # Ensure shapes are correct
        assert x.shape == (4, 1), f"Shape mismatch: x is {x.shape}"
        assert u.shape == (1, 1), f"Shape mismatch: u is {u.shape}"
        
        # Add scalar values to the cost
        cost += float(x.T @ Q @ x) + float(u.T @ R @ u)
    return cost


def solve_mpc(x0):
    U0 = np.zeros(N)  # Initial control guess
    bounds = [(-10, 10)] * N  # Control force limits
    res = minimize(mpc_cost, U0, args=(x0,), bounds=bounds, tol=1e-4)
    return res.x if res.success else U0

# Simulation settings
T = 200  # Total simulation steps
x = np.array([0, 0, np.pi * 0.1, 0])  # Initial state: [position, velocity, angle, angular velocity]

# Store results
x_log = []
u_log = []

# Run the simulation
for t in range(T):
    x_log.append(x.flatten())  # Ensure shape is (4,)
    #x_log.append(x)
    U_opt = solve_mpc(x)  # Get optimal control
    u = U_opt[0]          # Apply the first control action
    u_log.append(u)
    x = cartpole_dynamics(x, u)  # Evolve the system


x_log = np.array(x_log)
u_log = np.array(u_log)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(x_log[:, 0], label='Cart Position')
plt.plot(x_log[:, 2], label='Pole Angle')
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.title('Cart-Pole State Evolution, emphasize theta 50 position 10 in Q')

plt.subplot(2, 1, 2)
plt.plot(u_log, label='Control Input (Force)')
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.xlabel('Time Step')

plt.tight_layout()
plt.show()
