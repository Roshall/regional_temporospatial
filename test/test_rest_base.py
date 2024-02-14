from configs import cfg
from indices import build_tempo_spatial_index
from search.base import absorb, BaseSliding, base_search
from search.rest import expend
from search.verifier import obj_verify
from test.index_test_helper import grid_spt_tempo_idx_fake_data, query4test
from utilities.trajectory import Trajectory, TrajectoryIntervalSeg
from functools import partial


def test_absorb():
    segs = [[9, 20, 1, 5, 29, 30, 6, 9, 20, 23, 27, 28, 26, 27, 28, 29],
            [5, 6, 3, 5, 8, 13]]
    ans = [[1, 5], [6, 23], [26, 30], [8, 13]]
    trajs = {i: Trajectory(i, 0, seg) for i, seg in enumerate(segs)}
    res = [[traj.begin, traj.begin + traj.len] for traj in absorb(trajs, 4)]
    assert res == ans


class TestBaseSliding(object):
    traj_len = [[1, 21], [1, 5], [2, 10], [5, 16], [15, 20], [15, 21]]
    trajs = [TrajectoryIntervalSeg(i, itv[0], i & 1, itv[1] - itv[0]) for i, itv in enumerate(traj_len)]

    interval = 0, 20
    duration = 4
    labels = {0: 1, 1: 1}
    label_verifier = partial(obj_verify, labels)
    sliding = BaseSliding(trajs, interval, duration, label_verifier)

    def test_base_sliding(self):
        ans = [((0, 1), [1, 4]), ((0, 2, 3), [5, 8]), ((0, 2, 3), [6, 9]), ((0, 3), [10, 13]),
               ((0, 3), [12, 15]), ((0, 4, 5), [15, 18]), ((0, 4, 5), [17, 20])]
        res = [(tuple(pat.labels), pat.interval) for pat in self.sliding]
        assert res == ans

    def test_final(self):
        ans = [((0,1), [1,4]), ((0, 2, 3), [5, 9]), ((0, 3), [5, 15]), ((0, 4, 5), [15, 20])]
        res = [(tuple(pat.labels), pat.interval) for pat in expend(self.sliding, self.label_verifier)]
        assert res == ans


def test_base_search():
    cfg.merge_from_file('../configs/region_base.yml')
    spt_tempo = grid_spt_tempo_idx_fake_data(cfg)

    test1, test2 = query4test()
    res = [(frozenset(pat.labels), *pat.interval) for pat in base_search(spt_tempo, *test1)]
    assert res == [(frozenset([2, 3, 1]), 5, 10), (frozenset([3, 1]), 4, 10), (frozenset([1]), 1, 10)]

    res = [(frozenset(pat.labels), *pat.interval) for pat in base_search(spt_tempo, *test2)]
    assert res == [(frozenset([5, 1, 3]), 9, 18), (frozenset([2, 4, 5, 3]), 11, 20), (frozenset([2, 4, 3]), 11, 22)]

