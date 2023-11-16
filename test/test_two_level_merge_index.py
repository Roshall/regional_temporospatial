from functools import partial

from btree import BtreeMap, BtreeMultiMap

from indices.region import GridRegion
from test.test_helper import *
from indices.two_level_merge_index import TwoLevelMergeIndex, fuzzy


class TestIndices:
    Out3D = partial(TwoLevelMergeIndex, GridRegion, BtreeMap)
    Tempo2D = partial(TwoLevelMergeIndex, BtreeMap, BtreeMultiMap)
    Tempo2D_fuzzy = partial(fuzzy(TwoLevelMergeIndex), BtreeMap, BtreeMultiMap)
    fuzzy_user = TwoLevelMergeIndex(Out3D, Tempo2D_fuzzy)

    def test_out3d(self):
        out3d = self.Out3D()
        for key, obj, in zip(keys, objs):
            out3d.add(key, obj)
            assert out3d.where_contain(key) is obj

        res = list(out3d.where_intersect([(20, 30, 16, 70), (10, 100)]))
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
        res = list(user_idx.where_intersect([((15, 30, 16, 70), (10, 110)), ((2, 100), (1, 100))]))
        assert len(res) == 3

    def test_fuzzy(self):
        for *key, entry in zip(keys, zip(durations, begin), content):
            self.fuzzy_user.add(key, entry)
        res = list(self.fuzzy_user.where_intersect([((15, 30, 16, 70), (10, 110)), (46, 67)]))
        assert len(res) == 3

    def test_perhaps_intersect(self):
        for *key, entry in zip(keys, zip(durations, begin), content):
            self.fuzzy_user.add(key, entry)
        candidates, probation = self.fuzzy_user.perhaps_intersect([((15, 30, 16, 70), (10, 110)), (46, 67)])
        assert len(list(candidates)) == 0
        assert len(list(probation)) == 3
