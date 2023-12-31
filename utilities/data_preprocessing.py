import numpy as np

from utilities.trajectory import RawTraj, TrajTrack


def load_data(tracks, cols_name: list, fps, scale=100):
    """
    load raw scv data to trajectory entry.
    :param tracks: pandas data frame of tracks
    :param cols_name: [track_id, cls_id, frame_id, x, y, cls_id] are wanted, supply their actual properties name in order.
    :param fps: integer
    :param scale: the unit of our coordinate is 'cm', tell us how we scale raw (x, y).
    :return: trajectories tuple with format ('TrajTrace', 'tId start_frame track') and points of all trajectories.
    """
    tracks = tracks[cols_name]
    frames = tracks[cols_name[1]]
    lifelong = frames.max() - frames.min() + 1
    X = tracks[cols_name[2]]
    Y = tracks[cols_name[3]]
    bbox = np.array((X.min(), X.max(), Y.min(), Y.max())) * scale
    bbox = bbox.astype(np.int32)
    traces_by_id = tracks.groupby(cols_name[0])
    traj_ls = []
    for tid, t in traces_by_id:
        start_frame, cls_id = t[cols_name[1:3]].iloc[0]
        trajs = (t[cols_name[-2:]].to_numpy() * scale).astype(np.int32)
        traj_ls.append(TrajTrack(tid, cls_id,  start_frame, trajs))
    return RawTraj(fps, lifelong, bbox, traj_ls), tracks[cols_name[2:-1]].to_numpy()


def gen_border(bbox, x_num, y_num):
    xmin, xmax, ymin, ymax = bbox
    x_series = np.linspace(xmin, xmax, x_num, dtype=int)
    y_series = np.linspace(ymin, ymax, y_num, dtype=int)
    return x_series, y_series


def draw_traj_point_in_grid(data, reg_borders):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    fig, axs = plt.subplots()
    axs.scatter(*data.T)
    axs.grid()
    axs.xaxis.set_major_locator(FixedLocator(reg_borders[0]))
    axs.yaxis.set_major_locator(FixedLocator(reg_borders[1]))
    plt.show()


if __name__ == '__main__':
    import pandas as pd
    file_name = '../resource/00_tracks.csv'
    fps = 25
    cols = ['trackId', 'frame', 'xCenter', 'yCenter']
    raw_traj, raw_tracks = load_data(pd.read_csv(file_name), cols, fps)
    draw_traj_point_in_grid(raw_tracks, gen_border(raw_traj.bbox, 10, 15))
