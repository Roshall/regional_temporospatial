import numpy as np

from utilities.trajectory import RawTraj, TrajTrack


def traj_data(tracks, cols_name: list, fps, label_map, scale=100):
    """
    load raw scv data to trajectory entry.
    :param tracks: pandas data frame of tracks
    :param cols_name: [track_id, frame_id, x, y] are wanted, supply their actual properties name in order.
    :param fps: integer
    :param label_map: map traj id to its label
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
        cls_id = label_map[tid]
        start_frame = t[cols_name[1]].iat[0]
        trajs = (t[cols_name[-2:]].to_numpy() * scale).astype(np.int32)
        traj_ls.append(TrajTrack(tid, cls_id,  start_frame, trajs))
    return RawTraj(fps, lifelong, bbox, traj_ls)


def gen_border(bbox, x_num, y_num):
    xmin, xmax, ymin, ymax = bbox
    x_series = np.linspace(xmin, xmax, x_num, dtype=int)
    x_series = np.append(x_series, x_series[-1]+1)
    y_series = np.linspace(ymin, ymax, y_num, dtype=int)
    y_series = np.append(y_series, y_series[-1]+1)
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
    import dataset
    import os

    file_path = os.path.join(os.path.dirname(__file__), '../resource')
    fps = 25
    cls_map, data = dataset.load_rounD(file_path, '00')
    cols = ['trackId', 'frame', 'xCenter', 'yCenter']
    raw_traj = traj_data(data, cols, fps, cls_map)
    XY = data[cols[-2:]].to_numpy()
    draw_traj_point_in_grid(XY, gen_border(raw_traj.bbox, 10, 15))
