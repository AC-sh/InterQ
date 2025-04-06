import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import params

# Import system parameters
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------
# Q-Network (Function Approximation)
# -----------------------------------

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

lr = 0.01  # Learning rate
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 16
memory_size = 1000
num_episodes = 1800

# Initialize Q-network & optimizer
q_network = QNetwork(input_dim=2, output_dim=2).to(device)  # Input: [e1, e2], Output: [Q(no comm), Q(comm)]
target_q_network = QNetwork(input_dim=2, output_dim=2).to(device)
target_q_network.load_state_dict(q_network.state_dict())
target_q_network.eval()  # Target network is fixed

optimizer = optim.Adam(q_network.parameters(), lr=lr)
memory = []

rewards_per_episode = []
loss_per_episode = []


# epsilon-greedy policy
def eps_greedy_policy(epsilon, state_tensor):
  if np.random.rand() < epsilon:
    action = np.random.choice([0, 1])  # Random action (exploration)
  else:
    with torch.no_grad():
      q_values = q_network(state_tensor)
      action = torch.argmax(q_values).item()  # Greedy action (exploitation)

  return action


# ---------------
# Training Loop
# ---------------

for episode in range(num_episodes):
  error = np.random.multivariate_normal([0.0, 0.0], 2 * W).reshape(-1, 1)
  episode_reward = 0
  ep_loss = 0.0

  for k in range(N):
    state = np.array([error[0, 0], error[1, 0]])
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # Select action using epsilon-greedy policy
    action = eps_greedy_policy(epsilon, state_tensor)
    comm_cost = action * lambda_comm

    # Compute control
    #u = -K @ x_hat

    # Compute reward (negative cost)
    quadratic_cost = (Gamma * error.T @ (K.T @ K) @ error).item()
    reward = - (quadratic_cost + comm_cost)
    episode_reward = gamma * episode_reward + reward

    w = ((np.random.rand(1, 2) * 2 - 1) * 0.5).reshape(-1, 1)
    error = (1 - action) * (A @ error + w)

    # Store transition in memory
    next_state = np.array([error[0, 0], error[1, 0]])
    memory.append((state, action, reward, next_state))

    # Keep memory size bounded
    if len(memory) > memory_size:
      memory.pop(0)

    # Train Q-network (experience replay)
    if len(memory) >= batch_size:
      batch = random.sample(memory, batch_size)
      state_batch = torch.tensor([s[0] for s in batch], dtype=torch.float32, device=device)
      action_batch = torch.tensor([s[1] for s in batch], dtype=torch.int64, device=device)
      reward_batch = torch.tensor([s[2] for s in batch], dtype=torch.float32, device=device)
      next_state_batch = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=device)

      q_values = q_network(state_batch)
      q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
      next_q_values = target_q_network(next_state_batch).max(1)[0].detach()

      if k == N - 1:
        target_q_values = reward_batch
      else:
        target_q_values = reward_batch + gamma * next_q_values

      loss = nn.functional.huber_loss(q_values, target_q_values)
      ep_loss = ep_loss + loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      for param in q_network.parameters():
        param.grad.data.clamp_(-50, 50)

  if episode % 50 == 0:
    target_q_network.load_state_dict(q_network.state_dict())

  ep_loss = ep_loss / float(N)
  print(f'Training loss after epoch ----{episode} is {ep_loss:.3f}------')

        # Reduce exploration rate

  epsilon = max(epsilon_min, epsilon * epsilon_decay)
  print(f"Episode {episode+1}, Total Reward: {episode_reward:.2f}")

  rewards_per_episode.append(episode_reward)
  loss_per_episode.append(ep_loss)

# Save trained Q-network
torch.save(q_network.state_dict(), "q_network.pth")
print("Training Complete!")




