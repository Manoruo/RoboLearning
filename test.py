import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to TensorBoard logs (UPDATE THIS)
log_dir = "/home/mikea/Documents/Class/RoboLearning/hw1/data"

# Get event file (assuming one file in the directory)
event_file = [f for f in os.listdir(log_dir) if "tfevents" in f][0]
event_path = os.path.join(log_dir, event_file)

# Load event data
event_acc = EventAccumulator(event_path)
event_acc.Reload()

# Function to extract scalar data
def extract_scalar_data(tag):
    events = event_acc.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values

# Extract Behavioral Cloning metrics
bc_steps, bc_avg_return = extract_scalar_data("Eval_AverageReturn")
_, bc_std_return = extract_scalar_data("Eval_StdReturn")

# Extract Expert metrics
expert_steps, expert_avg_return = extract_scalar_data("Train_AverageReturn")
_, expert_std_return = extract_scalar_data("Train_StdReturn")

# Extract DAgger learning curve (assume it follows BC evaluation steps)
dagger_steps = bc_steps
dagger_avg_return = bc_avg_return  # This assumes DAgger continues from BC
dagger_std_return = bc_std_return

# Plot the learning curve
plt.errorbar(dagger_steps, dagger_avg_return, yerr=dagger_std_return, fmt='-o', capsize=5, label="DAgger")
plt.axhline(y=np.mean(expert_avg_return), color='r', linestyle='--', label="Expert Policy")
plt.axhline(y=np.mean(bc_avg_return), color='g', linestyle='--', label="Behavioral Cloning")

plt.xlabel("DAgger Iterations")
plt.ylabel("Mean Return")
plt.title("DAgger Learning Curve")
plt.legend()
plt.grid()
plt.savefig('my_plot.png')

