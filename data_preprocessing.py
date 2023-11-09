import pandas as pd
from trajectory import RawTraj, TrajTrack

file_name = '00_tracks.csv'
fps = 25

traces = pd.read_csv(file_name)
traces = traces[['trackId', 'frame', 'xCenter', 'yCenter']]
lifelong = traces.frame.max() - traces.frame.min() + 1
xmin, ymin, xmax, ymax = traces.xCenter.min(), traces.yCenter.min(), traces.xCenter.max(), traces.yCenter.max()
traces_by_id = traces.groupby('trackId')
traj_ls = []
for tid, t in traces_by_id:
    start_frame = t['frame'].iat[0]
    traj_ls.append(TrajTrack(tid, start_frame, t[['xCenter', 'yCenter']].to_numpy()))
raw_traj = RawTraj(fps, lifelong, (xmin, ymin, xmax, ymax), traj_ls)
# print(raw_traj.bbox)

# plot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

x_series = np.linspace(xmin, xmax, 10)
y_series = np.linspace(ymin, ymax, 10)
data = traces[['xCenter', 'yCenter']].to_numpy()
fig, axs = plt.subplots()
axs.scatter(*data.T)
axs.borders()
axs.xaxis.set_major_locator(FixedLocator(x_series))
axs.yaxis.set_major_locator(FixedLocator(y_series))
plt.show()
