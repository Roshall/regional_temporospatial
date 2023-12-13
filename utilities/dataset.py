from zipfile import ZipFile

import pandas as pd


def load_rounD(path, track_id):
    with ZipFile(f'{path}/rounD.zip') as zf:
        prefix = f'rounD/data/{track_id}_tracks'
        with zf.open(f'{prefix}Meta.csv') as f:
            meta = pd.read_csv(f)

        with zf.open(f'{prefix}.csv') as f:
            data = pd.read_csv(f)
    desired_cols = ['trackId, class']
    class_map = meta[desired_cols]
    label = ('car', 'truck', 'van', 'bus', 'trailer', 'bicycle', 'motorcycle', 'pedestrian')
    label_map = dict((l, i) for i, l in enumerate(label))
    class_map = class_map['class'].transform(lambda x: label_map[x])
    class_map = {i: row['class'] for i, row in class_map.iterrows()}
    return class_map, data


def load_fake():
    import os
    file_name = os.path.join(os.path.dirname(__file__), '../resource/fake_tracks.csv')
    data = pd.read_csv(file_name)
    cls = [0, 0, 0, 1, 1]
    cls_map = dict(enumerate(cls, 1))
    return cls_map, data
