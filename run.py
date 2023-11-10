import numpy as np

from config import config
from data_preprocessing import gen_border
from indices import UserIdx
from region import GridRegion
from trajectory import NaiveTrajectorySeg


def build_tempo_spatial_index(trajs):
    # 1. find trajectory's region
    # config.gird_border = gen_border(trajs.bbox, 10, 15)
    reg = GridRegion()
    user_idx = UserIdx()

    # we need to fill the territory with distinct objects
    class _C:
        def __init__(self, arg):
            pass
    fill = np.vectorize(_C)
    reg.territory = fill(reg.territory)

    for tid, beg, track in trajs.traj_track:
        track_iter = iter(track)
        first_reg = reg.where_contain(next(track_iter))
        track_life = len(track)
        start, end = 0, 1
        # if some point jumps to another region, we cut the trajectory to a new segment.
        for pid, coord in enumerate(track_iter, 1):
            if (last_reg := reg.where_contain(coord)) is not first_reg:
                # 2. insert to index
                user_idx.add(((track_life, track[start]), (end - start, start + beg)),
                             NaiveTrajectorySeg(tid, track[start:end], True))
                start = end
                first_reg = last_reg
            end += 1

        user_idx.add(((track_life, track[start]), (end - start, start + beg)),
                     NaiveTrajectorySeg(tid, track[start:end]))
    return user_idx


def search(tempo_spat_idx, region, labels, duration, interval):
    tempo_spat_idx.query(region, labels, duration)
