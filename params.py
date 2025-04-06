import numpy as np

# -----------------------------
# System Dynamics & Cost Setup
# -----------------------------
A = np.array([[1.5, 2.0], [0.0, 1.51]])
B = np.array([[0.0], [1.0]])
Q = np.array([[1.0, 0.0], [0.0, 1.0]])
R = np.array([1.0])
W = np.array([[0.5, 0.0], [0.0, 0.5]])
lambda_comm = 60
N = 50  # Rollout horizon
gamma = 0.95  # Discount factor
