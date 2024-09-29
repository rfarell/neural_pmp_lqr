# models.py

import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, T):
        super(PolicyNetwork, self).__init__()
        self.T = T
        # Simple linear layers for each time step
        self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(T)])

    def forward(self, q0):
        batch_size = q0.size(0)
        u_seq = []
        q = q0

        for layer in self.layers:
            u = layer(q)
            u_seq.append(u)
            # For the purpose of the policy network, we don't update q here

        u_seq = torch.stack(u_seq, dim=1)  # Shape: [batch_size, T, 1]
        return u_seq