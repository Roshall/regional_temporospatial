from functools import partial

import pandas as pd

from naive import NaiveSliding
import helper as f
from utilities.box2D import Box2D


def data_by_frame(filename):
    data = pd.read_pickle(filename)
    return data.sort_values(by='fid').groupby('fid')  # in case that groups are not sorted by fid


if __name__ == '__main__':
    data_path = '/home/lg/VDBM/spatiotemporal/resource/dataset/archeday_0410_s.pkl'
    frames = data_by_frame(data_path)
    region = Box2D((0, 3800, 0, 2100))
    labels = {0: 1}
    duration = 2
    obj_verifier = partial(f.obj_verify, labels)

    sliding = NaiveSliding(duration, region.enclose, obj_verifier, f.len_verify, tuple(labels))
    res = f.expend(sliding, obj_verifier)


