import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import params
import math

A = params.A
B = params.B
Q = params.Q
R = params.R
W = params.W
lambda_comm = params.lambda_comm
N = params.N
gamma = params.gamma

A_discounted = math.sqrt(gamma) * A
R_discounted = params.R / params.gamma

# ---------------------------------------
# Solving for Algebraic Riccati Equation
# ---------------------------------------
P = scipy.linalg.solve_discrete_are(A_discounted, B, Q, R_discounted)
Gamma = R + gamma * B.T @ P @ B
K = gamma * np.linalg.inv(Gamma) @ (B.T @ P @ A)

# -----------------------------
# Load Trained Q-Network
# -----------------------------

class QNetwork(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(QNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dim, 100)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, 100)
    self.fc4 = nn.Linear(100, output_dim)

  def forward(self, x):
    x = torch.nn.functional.gelu(self.fc1(x))
    x = torch.nn.functional.gelu(self.fc2(x))
    x = torch.nn.functional.gelu(self.fc3(x))
    return self.fc4(x)

q_network = QNetwork(input_dim=2, output_dim=2)
q_network.load_state_dict(torch.load("q_network.pth"))
q_network.eval()

# -----------------------------
# Simulation with Learned Policy
# -----------------------------
n_samples = 50
noise = np.zeros((n_samples, N, 2))
initial = np.zeros((n_samples,1,2))

for k in range(n_samples):
  #noise[k, :, :] = np.random.multivariate_normal([0.0, 0.0], 2 * W, size=N)
  noise[k, :, :] = (np.random.rand(N,2) * 2 - 1) * 0.5
  initial = np.random.multivariate_normal([0.0, 0.0], 2 * W).reshape(-1, 1)

ep_control_cost = 0.0
ep_sch_cost = 0.0
sum_len = 0.0

for num in range(n_samples):

  trajectory_true = []
  trajectory_est = []
  communication_instants = []
  control_cost = 0.0
  scheduling_cost = 0.0
  x_true = np.ones((2,1))
  x_hat = np.zeros((2,1))  # Initial estimate

  for k in range(N):
    # State vector for RL agent
    state = np.array([x_true[0, 0] - x_hat[0, 0], x_true[1, 0] - x_hat[1, 0]])
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # Choose action using trained Q-network
    with torch.no_grad():
      q_values = q_network(state_tensor)
      action = torch.argmax(q_values).item()  # Greedy action

    # Communication decision
    if action == 1:
      x_hat = np.copy(x_true)
      communication_instants.append(k)

    # Compute control
    u = -K @ x_hat

    control_cost = control_cost * gamma + (x_true.T @ Q @ x_true + u.T @ R @ u).item()
    scheduling_cost = scheduling_cost * gamma + lambda_comm * action

    trajectory_true.append(x_true)
    trajectory_est.append(x_hat)

    # Apply control & system noise
    w = noise[num, k, :].reshape(-1, 1)
    x_true = A @ x_true + B @ u + w

    # Predict next state estimate
    x_hat = A @ x_hat + B @ u

  # Convert trajectory to readable format
  trajectory_true = np.hstack(trajectory_true)
  trajectory_est = np.hstack(trajectory_est)
  sum_len += len(communication_instants) / N

  ep_control_cost += control_cost
  ep_sch_cost += scheduling_cost

avg_period = sum_len / n_samples

print("{:.2e}".format(ep_control_cost / n_samples))
print("{:.2e}".format(ep_sch_cost / n_samples))
print(avg_period)


