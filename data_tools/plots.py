# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
strong_scaling = pd.read_csv('groundtruth_strong_scaling.txt', sep=' ')
strong_scaling['time'] = strong_scaling['end'] - strong_scaling['start']
strong_scaling.sort_values('nproc', inplace=True)

# only using gist values
strong_scaling = strong_scaling[strong_scaling['dataset'] == 'gist']

strong_scaling

# %%
strong_scaling['bpps'] = 1000000 / strong_scaling['time']

# %%
original_strong_scaling = strong_scaling[strong_scaling['q_block'] == 1]
blocked_strong_scaling = strong_scaling[strong_scaling['q_block'] == 100]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(original_strong_scaling['nproc'], original_strong_scaling['bpps'], label='Original')
ax[0].plot(blocked_strong_scaling['nproc'], blocked_strong_scaling['bpps'], label='Blocked')
ax[0].set_xlabel('Number of threads')
ax[0].set_ylabel('Base points per second')
ax[0].legend()

ax[1].plot(original_strong_scaling['nproc'], original_strong_scaling['time'], label='Original')
ax[1].plot(blocked_strong_scaling['nproc'], blocked_strong_scaling['time'], label='Blocked')
ax[1].set_xlabel('Number of threads')
ax[1].set_ylabel('Time (s)')


fig.tight_layout()
fig.savefig('plots/strong_scaling.png', dpi=300)

plt.show()

# %%
# load scaling data
scaling = pd.read_csv('groundtruth_scaling.txt', sep=' ')
scaling['time'] = scaling['end'] - scaling['start']


scaling
# %%
scaling['bpps'] = scaling['data_size'] / scaling['time']

# %%
original_scaling = scaling[scaling['q_block'] == 1]
blocked_scaling = scaling[scaling['q_block'] == 100]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(original_scaling['data_size'], original_scaling['bpps'], label='Original')
ax[0].plot(blocked_scaling['data_size'], blocked_scaling['bpps'], label='Blocked')
ax[0].set_xlabel('Base points')
ax[0].set_ylabel('Base points per second')
ax[0].legend()

ax[1].plot(original_scaling['data_size'], original_scaling['time'], label='Original')
ax[1].plot(blocked_scaling['data_size'], blocked_scaling['time'], label='Blocked')
ax[1].set_xlabel('Base points')
ax[1].set_ylabel('Time (s)')

fig.tight_layout()
fig.savefig('plots/scaling.png', dpi=300)

plt.show()

# %%
# subplots of the bpps of the previous two plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(original_scaling['data_size'], original_scaling['bpps'], label='Original')
ax[0].plot(blocked_scaling['data_size'], blocked_scaling['bpps'], label='Blocked')
ax[0].set_xlabel('Base points')
ax[0].set_ylabel('Base points per second')
ax[0].legend()
ax[0].set_title('Scaling')
# ax[0].set_xscale('log')

ax[1].plot(original_strong_scaling['nproc'], original_strong_scaling['bpps'], label='Original')
ax[1].plot(blocked_strong_scaling['nproc'], blocked_strong_scaling['bpps'], label='Blocked')
ax[1].set_xlabel('Number of threads')
ax[1].set_ylabel('Base points per second')
ax[1].legend()
ax[1].set_title('Strong Scaling')

fig.tight_layout()
fig.savefig('plots/bpps.png', dpi=300)

plt.show()
# %%
# grid of block sizes
grid_values = pd.read_csv('groundtruth_block_grid_serial_outer.txt', sep=' ')
grid_values['time'] = grid_values['end'] - grid_values['start']
grid_values['bpps'] = grid_values['data_size'] / grid_values['time']
grid_values
# %%
# heatmap of runtimes vs block sizes

pivot = grid_values.pivot(columns='b_block', index='q_block', values='time')

fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(pivot.values, cmap='viridis', interpolation='nearest', aspect='auto')

# add color bar
fig.colorbar(cax)

# set x and y ticks
x_labels = pivot.columns
y_labels = pivot.index

ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(y_labels)))
ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

# rotate x-axis labels if needed
plt.xticks(rotation=90)

# add annotations (optional)
data = pivot.values
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='white', rotation=-45)

# axis labels
ax.set_xlabel('Data block size')
ax.set_ylabel('Query block size')

plt.tight_layout()
fig.savefig('plots/block_grid_heatmap.png', dpi=300)

plt.show()
# %%
