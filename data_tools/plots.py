# %%
import pandas as pd
import matplotlib.pyplot as plt

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
original = strong_scaling[strong_scaling['q_block'] == 1]
blocked = strong_scaling[strong_scaling['q_block'] == 100]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(original['nproc'], original['bpps'], label='Original')
ax[0].plot(blocked['nproc'], blocked['bpps'], label='Blocked')
ax[0].set_xlabel('Number of threads')
ax[0].set_ylabel('Base points per second')
ax[0].legend()

ax[1].plot(original['nproc'], original['time'], label='Original')
ax[1].plot(blocked['nproc'], blocked['time'], label='Blocked')
ax[1].set_xlabel('Number of threads')
ax[1].set_ylabel('Time (s)')

plt.show()

# %%
# load scaling data
scaling = pd.read_csv('groundtruth_scaling.txt', sep=' ')
scaling['time'] = scaling['end'] - scaling['start']


scaling
# %%
scaling['bpps'] = scaling['data_size'] / scaling['time']

# %%
original = scaling[scaling['q_block'] == 1]
blocked = scaling[scaling['q_block'] == 100]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(original['data_size'], original['bpps'], label='Original')
ax[0].plot(blocked['data_size'], blocked['bpps'], label='Blocked')
ax[0].set_xlabel('Base points')
ax[0].set_ylabel('Base points per second')
ax[0].legend()

ax[1].plot(original['data_size'], original['time'], label='Original')
ax[1].plot(blocked['data_size'], blocked['time'], label='Blocked')
ax[1].set_xlabel('Base points')
ax[1].set_ylabel('Time (s)')

plt.show()

# %%
