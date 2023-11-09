from functools import partial

from trajectory import Traj_Meta
from two_level_merge_index import TwoLevelMergeIndex
from btree import BtreeMap, BtreeMultiMap
from region import GridRegion
from test_helper import *


class TestIndices:
    Out3D = partial(TwoLevelMergeIndex, BtreeMap, GridRegion)
    Tempo2D = partial(TwoLevelMergeIndex, BtreeMap, BtreeMultiMap)

    def test_out3d(self):
        out3d = self.Out3D()
        for key, obj, in zip(keys, objs):
            out3d.add(key, obj)
            assert out3d.where_contain(key) is obj

        res = list(out3d.where_intersect([(10, 100), (20, 30, 16, 70)]))
        assert len(res) == 4

    def test_tempo2D(self):
        tempo2d = self.Tempo2D()

        for *key, entry in zip(durations, begin, content):
            tempo2d.add(key, entry)
            assert tempo2d.where_contain(key) == entry

        res = list(tempo2d.where_intersect([(2,100),(1,100)]))
        assert len(res) == 2

    def test_user_idx(self):
        user_idx = TwoLevelMergeIndex(self.Out3D, self.Tempo2D)
        for *key, entry in zip(keys, zip(durations, begin), content):
            user_idx.add(key, entry)
            assert user_idx.where_contain(key) == entry
        # search
        res = list(user_idx.where_intersect([((10, 110), (15, 30, 16, 70)), ((2,100),(1,100))]))
        assert len(res) == 1
