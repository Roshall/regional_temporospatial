from zipfile import ZipFile

import pandas as pd


def load_rounD(path, track_id):
    with ZipFile(f'{path}/rounD.zip') as zf:
        prefix = f'rounD/data/{track_id}_tracks'
        with zf.open(f'{prefix}Meta.csv') as f:
            meta = pd.read_csv(f, index_col='trackId', usecols=['trackId', 'class'])

        with zf.open(f'{prefix}.csv') as f:
            data = pd.read_csv(f)
    label = ('car', 'truck', 'van', 'bus', 'trailer', 'bicycle', 'motorcycle', 'pedestrian')
    class_map = meta['class'].replace({l: i for i, l in enumerate(label)})
    return class_map, data


def load_fake():
    import os
    file_name = os.path.join(os.path.dirname(__file__), '../resource/fake_tracks.csv')
    data = pd.read_csv(file_name)
    cls = [0, 0, 0, 1, 1]
    cls_map = dict(enumerate(cls, 1))
    return cls_map, data
