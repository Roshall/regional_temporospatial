import numpy as np
import pandas as pd

from run import build_tempo_spatial_index
from utilities.config import config
from utilities.data_preprocessing import load_data, gen_border
from utilities.trajectory import NaiveTrajectorySeg


class TestBuildTempSpatialIndex:
    trajs, _ = load_data(pd.read_csv('../resource/fake_tracks.csv'), ['tid', 'frameId', 'x', 'y'], 1)
    broders = gen_border(trajs.bbox, 5, 6)
    config.gird_border = broders
    tempo_spatial = build_tempo_spatial_index(trajs)

    def test_build_tempo_spatial_index(self):
        res = self.tempo_spatial.where_intersect([((10, 30), (0, 1, 0, 1)), (0, 10)])
        for entry in res:
            match entry:
                case NaiveTrajectorySeg(2, seg, True):
                    ans = np.array([[0, 20], [20, 30], [60, 50], [90, 78]])
                    assert (seg == ans).all()
                case NaiveTrajectorySeg(1, seg, True):
                    ans = np.array([[0, 0], [20, 10], [50, 13], [70, 40]])
                    assert (seg == ans).all()
                case _:
                    assert False
