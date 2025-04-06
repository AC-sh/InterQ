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
N = 1000  # Time steps
error = np.zeros((N, 2))
actions = []

for num in range(N):
  err = np.random.multivariate_normal([0.0, 0.0], 2 * W).reshape(-1, 1)
  error[num,:] = err.reshape(1, -1)
  state = np.array([err[0, 0], err[1, 0]])
  state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # Choose action using trained Q-network

  with torch.no_grad():
    q_values = q_network(state_tensor)
    action = torch.argmax(q_values).item()  # Greedy action
  actions.append(action)

colors = []
for act in actions:
  color = 'red' if act == 1 else 'blue'
  colors.append(color)

plt.figure(figsize=(6,6))

for i, v in enumerate(error):
  x, y = v[0], v[1]
  plt.scatter(x, y, edgecolor=colors[i], facecolor='none', s=100)

a = 2.1  # Semi-major axis
b = 0.4  # Semi-minor axis
theta = np.linspace(0, 2 * np.pi, 300)  # Angle range

# Parametric equations for the ellipse
xx = a * np.cos(theta)
yy = b * np.sin(theta)

angle = np.radians(-13)

# Rotation matrix
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])

# Apply rotation
xy_rotated = np.dot(R, np.vstack((xx, yy)))
x_rotated, y_rotated = xy_rotated[0, :], xy_rotated[1, :]

# Plot the ellipse
plt.plot(x_rotated, y_rotated)

plt.show()



