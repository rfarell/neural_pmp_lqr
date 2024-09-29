# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import PolicyNetwork
from utils import plot_loss_curve, plot_trajectories

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LQR Problem Parameters
A = torch.tensor([[0.]], device=device)  # System dynamics matrix
B = torch.tensor([[1.]], device=device)  # Control matrix
R = torch.tensor([[1.]], device=device)  # State cost matrix
Q = torch.tensor([[0.]], device=device)  # Control cost matrix

# Time horizon and time steps
time_steps = [0, 1, 2, 4, 8]
dt = [t2 - t1 for t1, t2 in zip(time_steps[:-1], time_steps[1:])]
T = len(time_steps) - 1  # Number of intervals

# Training parameters
num_iterations = 1000
batch_size = 1600
learning_rate = 0.01

# Initialize policy network
policy_net = PolicyNetwork(T).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Lists to store loss values
train_loss_list = []

# Training loop
for iteration in range(num_iterations):
    # Generate random initial states q0
    q0 = torch.randn(batch_size, 1, device=device) * 10  # Initial positions

    # Step 1: Forward pass through the policy network to get control actions
    u_seq = policy_net(q0)  # Shape: [batch_size, T, 1]

    # Step 2: Simulate the dynamics to get state sequence
    q_seq = [q0]
    for k in range(T):
        q_prev = q_seq[-1]
        u_prev = u_seq[:, k, :]
        delta_t = dt[k]

        # Discrete dynamics: q_k = q_{k-1} + delta_t * (A q_{k-1} + B u_{k})
        f = A @ q_prev.T + B @ u_prev.T  # Shape: [1, batch_size]
        q_next = q_prev + delta_t * f.T
        q_seq.append(q_next)

    q_seq = torch.stack(q_seq, dim=1)  # Shape: [batch_size, T+1, 1]

    # Step 3: Compute adjoint variables p_k backwards
    # Initialize p_T = R q_T (since g(q_T) = 0)
    p_T = R @ q_seq[:, -1, :].permute(1, 0)
    p_seq = [p_T.T]

    for k in reversed(range(T)):
        q_k = q_seq[:, k+1, :]  # q_{k+1}
        u_k = u_seq[:, k, :]
        delta_t = dt[k]

        # Compute partial derivatives
        H_q = (R @ q_k.T + A.T @ p_seq[0].T).T  # Shape: [batch_size, 1]
        p_prev = p_seq[0] - delta_t * H_q  # p_{k} = p_{k+1} - delta_t * H_q
        p_seq.insert(0, p_prev)

    p_seq = torch.stack(p_seq, dim=1)  # Shape: [batch_size, T+1, 1]

    # Compute loss
    loss = 0.0
    for k in range(T):
        q_k = q_seq[:, k, :]
        u_k = u_seq[:, k, :]
        p_k = p_seq[:, k, :]

        # Hamiltonian partial derivative with respect to control
        H_u = (B.T @ p_k.T + Q @ u_k.T).T  # Shape: [batch_size, 1]
        loss += H_u.norm()

    loss = loss / batch_size

    # Backpropagation and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store and print loss
    train_loss_list.append(loss.item())
    if (iteration + 1) % 10 == 0:
        print(f'Iteration [{iteration + 1}/{num_iterations}], Loss: {loss.item():.4f}')

# Plot loss curve
plot_loss_curve(train_loss_list)

# Visualization of final model with multiple initial states
initial_positions = [-20., -10., 0., 10., 20.]
trajectories = []

for q0_val in initial_positions:
    q0_test = torch.tensor([[q0_val]], device=device)
    u_seq_test = policy_net(q0_test)
    q_seq_test = [q0_test]

    for k in range(T):
        q_prev = q_seq_test[-1]
        u_prev = u_seq_test[:, k, :]
        delta_t = dt[k]
        f = A @ q_prev.T + B @ u_prev.T
        q_next = q_prev + delta_t * f.T
        q_seq_test.append(q_next)

    q_seq_test = torch.stack(q_seq_test, dim=1).detach().cpu().numpy().flatten()
    u_seq_test = u_seq_test.detach().cpu().numpy().flatten()
    trajectories.append({
        'initial_position': q0_val,
        'positions': q_seq_test,
        'controls': u_seq_test
    })

# Plot trajectories for multiple initial positions
plot_trajectories(time_steps, trajectories)