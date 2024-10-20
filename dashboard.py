import numpy as np
import matplotlib.pyplot as plt

# Load violation data
violation_data = np.load('violation_data.npy', allow_pickle=True)

# Extract violation times
violation_times = [v['violation_time'] for v in violation_data]

# Calculate total number of violations
total_violations = len(violation_times)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Histogram for violation times
ax1.hist(violation_times, bins=30, color='blue', alpha=0.7)
ax1.set_title(f'Total Number of Violations Over Time: {total_violations}')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Number of Violations')
ax1.grid(axis='y')

# Line chart for cumulative violations
cumulative_violations = np.arange(1, total_violations + 1)
ax2.plot(violation_times, cumulative_violations, marker='o', color='orange')
ax2.set_title('Cumulative Number of Violations Over Time')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Cumulative Violations')
ax2.grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
