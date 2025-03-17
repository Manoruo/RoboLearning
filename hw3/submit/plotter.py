import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_section_results(file):
    """
    Extract training progress from TensorBoard logs.
    """
    X, Y, best_mean_rewards = [], [], []
    best_so_far = -np.inf

    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                print(v.simple_value)
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
                best_so_far = max(best_so_far, v.simple_value)
                best_mean_rewards.append(best_so_far)

    # Trim X to match length of Y and best rewards
    return np.array(X[1:]), np.array(Y), np.array(best_mean_rewards)

def plot_combined_learning_curve(logdirs, labels, save_path="combined_learning_curve.png"):
    """
    Plots a single graph averaging performance across three runs for both DQN and DDQN.
    - X-axis: Timesteps (log scale)
    - Y-axis: Average Per-Epoch Reward (solid lines) and Best Mean Reward So Far (dashed lines)
    """
    plt.figure(figsize=(10, 6))

    aggregated_data = {}

    for group in set(labels):  # Process separately for DQN & DDQN
        
        # Get files of similar groups (DQN or DDQN)
        group_logdirs = [logdir for logdir, label in zip(logdirs, labels) if label == group]

        # Record data for each group
        all_X, all_Y, all_best_rewards = [], [], []
        for logdir in group_logdirs:
            eventfiles = glob.glob(os.path.join(logdir, 'events*'))
            for eventfile in eventfiles:
                X, Y, best_rewards = get_section_results(eventfile)
                if len(X) == 0 or len(Y) == 0 or len(best_rewards) == 0:
                    continue  # Skip empty results
                all_X.append(X)
                all_Y.append(Y)
                all_best_rewards.append(best_rewards)

        if not all_X:  # No valid data found
            print(f"Warning: No valid data found for {group}")
            continue

        X_mean = np.mean(np.array(all_X), axis=0)
        Y_mean = np.mean(np.array(all_Y), axis=0)
        Y_std = np.std(np.array(all_Y), axis=0)
        best_mean_rewards = np.mean(np.array(all_best_rewards), axis=0)

        aggregated_data[group] = (X_mean, Y_mean, Y_std, best_mean_rewards)

    # Plot both DQN and DDQN
    for label, (X_mean, Y_mean, Y_std, best_mean_rewards) in aggregated_data.items():
        plt.plot(X_mean, Y_mean, label=f"{label} (Avg Return)", alpha=0.8)
        plt.fill_between(X_mean, Y_mean - Y_std, Y_mean + Y_std, alpha=0.2)
        #plt.plot(X_mean, best_mean_rewards, linestyle="dashed", label=f"{label} (Best So Far)")

    plt.xlabel("Timesteps (log scale)")
    plt.ylabel("Reward")
    plt.xscale("log")  # Scientific notation for x-axis
    plt.title("DQN vs. DDQN: Average Performance Across 3 Runs")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    
    logdirs = [
        os.path.join( "data", "q1_doubledqn_1_LunarLander-v3_17-03-2025_10-34-45"),
        os.path.join("data", "q1_doubledqn_2_LunarLander-v3_17-03-2025_10-44-19"),
        os.path.join("data", "q1_doubledqn_3_LunarLander-v3_17-03-2025_10-50-48"),
        os.path.join("data", "q1_dqn_1_LunarLander-v3_16-03-2025_21-49-59"),
        os.path.join("data", "q1_dqn_2_LunarLander-v3_16-03-2025_22-04-08"),
        os.path.join("data", "q1_dqn_3_LunarLander-v3_16-03-2025_22-18-37")
    ]

    labels = ["DDQN", "DDQN", "DDQN", "DQN", "DQN", "DQN"]

    plot_combined_learning_curve(logdirs, labels)
