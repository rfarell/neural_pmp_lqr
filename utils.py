# utils.py

import matplotlib.pyplot as plt

def plot_loss_curve(train_loss_list):
    plt.figure()
    plt.plot(train_loss_list, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('plots/loss_curve.png')
    plt.show()

def plot_trajectories(time_steps, trajectories):
    plt.figure(figsize=(12, 6))
    for traj in trajectories:
        plt.plot(time_steps, traj['positions'], marker='o', label=f"q0 = {traj['initial_position']}")
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position Trajectories for Different Initial Positions')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/position_trajectories.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    for traj in trajectories:
        plt.step(time_steps[:-1], traj['controls'], where='post', label=f"q0 = {traj['initial_position']}")
    plt.xlabel('Time')
    plt.ylabel('Control')
    plt.title('Control Trajectories for Different Initial Positions')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/control_trajectories.png')
    plt.show()