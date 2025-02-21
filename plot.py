import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
df = pd.read_csv('arm_poses_20250218_150324.csv')

# Scale the position and rotation values
scale = 1e6
df_scaled = df.copy()
position_cols = ['X', 'Y', 'Z']
rotation_cols = ['RX', 'RY', 'RZ']

for col in position_cols + rotation_cols:
    df_scaled[col] = df[col] / scale

# Create figure with adjusted size and grid
fig = plt.figure(figsize=(15, 12))
gs = plt.GridSpec(3, 2, figure=fig)

# 3D trajectory plot
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.plot(df_scaled['X'], df_scaled['Y'], df_scaled['Z'])
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D Trajectory')

# Position plots over time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df_scaled['X'], label='X')
ax2.plot(df_scaled['Y'], label='Y')
ax2.plot(df_scaled['Z'], label='Z')
ax2.set_title('Position over Time')
ax2.set_xlabel('Sample')
ax2.set_ylabel('Position (m)')
ax2.legend()

# Rotation plots over time
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df_scaled['RX'], label='RX')
ax3.plot(df_scaled['RY'], label='RY')
ax3.plot(df_scaled['RZ'], label='RZ')
ax3.set_title('Rotation over Time')
ax3.set_xlabel('Sample')
ax3.set_ylabel('Rotation (rad)')
ax3.legend()

# XY trajectory
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(df_scaled['X'], df_scaled['Y'])
ax4.set_title('XY Trajectory')
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.axis('equal')  # Make the plot aspect ratio 1:1

# Gripper state plot
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(df['grippers_angle']/scale, label='Angle', color='blue')
ax5.set_ylabel('Gripper Angle Distance (m)', color='blue')
ax5.tick_params(axis='y', labelcolor='blue')

# Add second y-axis for gripper effort
ax5_twin = ax5.twinx()
ax5_twin.plot(df['grippers_effort'], label='Effort', color='red', alpha=0.7)
ax5_twin.set_ylabel('Gripper Effort', color='red')
ax5_twin.tick_params(axis='y', labelcolor='red')

ax5.set_title('Gripper State over Time')
ax5.set_xlabel('Sample')

# Add combined legend
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()

# Print some basic statistics
print("\nStatistics:")
print(df.describe())