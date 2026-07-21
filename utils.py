import numpy as np
import matplotlib.pyplot as plt

def plot_reward_and_variance(rewards, window=100, save_path=None):
    rewards = np.array(rewards)
    episodes = np.arange(len(rewards))

    fig, ax1 = plt.subplots(figsize=(12, 4))

    # Reward
    ax1.plot(episodes, rewards, alpha=0.3, label='Reward')
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], ma, color='red',
                 label=f'Moving Avg ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Variance
    if len(rewards) >= window:
        ax2 = ax1.twinx()
        moving_std = [
            rewards[i:i+window].std()
            for i in range(len(rewards) - window + 1)
        ]
        ax2.plot(episodes[window-1:], moving_std, color='purple',
                 label='Reward Std')
        ax2.set_ylabel('Std Dev')
        ax2.legend(loc='upper right')

    plt.title('DQN Reward and Stability')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_dqn_loss(losses, window=100, save_path=None):
    losses = np.array(losses)

    plt.figure(figsize=(10, 4))
    plt.plot(losses, alpha=0.4, label='Loss per update')

    if len(losses) >= window:
        ma = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(ma, color='red', label=f'Moving Avg ({window})')

    plt.xlabel('Gradient update step')
    plt.ylabel('TD Loss')
    plt.title('DQN Training Loss')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
