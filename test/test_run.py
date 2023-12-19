import heapq

import numpy as np

from run import build_tempo_spatial_index, verify_seg, candidate_verified_queue, yield_co_move, \
    base_query, SequentialSearcher
from utilities.box2D import Box2D
from utilities.config import config
from utilities.data_preprocessing import traj_data, gen_border
from utilities.trajectory import NaiveTrajectorySeg, RunTimeTrajectorySeg, BasicTrajectorySeg
from utilities import dataset


class TestTempSpatialIndex:
    cls_map, data = dataset.load_fake()
    trajs = traj_data(data, ['tid', 'frameId', 'x', 'y'], 1, cls_map)
    broders = gen_border(trajs.bbox, 5, 6)
    config.gird_border = broders
    tempo_spatial = build_tempo_spatial_index(trajs)

    def test_build_tempo_spatial_index(self):
        cands, probs = self.tempo_spatial[0].where_intersect([((0, 1, 0, 1), (10, 30)), (0, 10)])
        for can in cands:
            for entry in can:
                match entry:
                    case NaiveTrajectorySeg(2, _, 0, seg):
                        ans = np.array([[0, 20], [20, 30], [60, 50], [90, 78]])
                        assert (seg == ans).all()
                    case NaiveTrajectorySeg(1, _, 0, seg):
                        ans = np.array([[0, 0], [20, 10], [50, 13], [70, 40]])
                        assert (seg == ans).all()
                    case _:
                        assert False

    def test_base_query(self):
        interval = 0, 10
        duration = 2, 30
        labels = {0: 1}
        region = Box2D((0, 400, 0, 500))

        res = list(base_query(self.tempo_spatial, region, labels, duration, interval))
        assert len(res) == 3


def test_verify_seg():
    region = Box2D((11, 111, 11, 111))
    # start and end are out
    points1 = np.array([[10, 67], [13, 83], [19, 92], [23, 113], [41, 121], [59, 102], [81, 83], [113, 75], [101, 49],
                        [54, 19], [31, 7], [27, 27], [19, 45], [13, 51], [9, 55]])
    seg1 = NaiveTrajectorySeg(1, 0, 0, points1)
    res = verify_seg(region, seg1, 3)
    assert len(res) == 1
    assert res == [RunTimeTrajectorySeg(1, 11, 0, 3)]
    res = verify_seg(region, seg1, 2)
    assert len(res) == 4
    assert [seg.begin for seg in res] == [1, 5, 8, 11]

    seg2 = NaiveTrajectorySeg(1, 0, 0, points1[1:-1])
    res = verify_seg(region, seg2, 3)
    assert len(res) == 2
    assert [(seg.begin, seg.len) for seg in res] == [(0, 2), (10, 3)]


def test_candidate_verified_queue():
    x = np.linspace(0, np.pi, 100)
    Y = [np.cos(x * 20) * 10, np.sin((x + np.pi / 4) * 20) * 10]
    all_points = [np.vsplit(np.vstack((x, y)).T, 10) for y in Y]
    cand = [[[NaiveTrajectorySeg(i, j * 10, 0, obj_ps) for j, obj_ps in enumerate(ind)] for i, ind in
             enumerate(all_points)]]

    region = Box2D((0, 1.5 * np.pi, 0, 10))
    data = list(candidate_verified_queue(region, cand, 5, 10))
    assert len(list(data)) == 30


def test_yield_co_move():
    duration = 5
    obj_info = [(1, 0), (4, 1), (8, 1), (9, 2), (10, 0), (11, 0)]
    obj_info = [BasicTrajectorySeg(i, *info)for i, info in enumerate(obj_info)]
    active_space = dict(enumerate(obj_info))
    ts = 15
    ids = [active_space[i] for i in (1, 2, 4)]

    label_quizzes = [{0: 2, 1: 2, 2: 1}, {0: 3, 1: 1, 2: 1}, {0: 1, 1: 1, 2: 0}]

    ans = [1, 0, 3]
    for labels, a in zip(label_quizzes, ans):
        test_map = dict(active_space)
        res = list(yield_co_move(duration, labels, test_map, ts, ids))
        assert len(res) == a


def test_sequential_search():
    traj1 = [(1, 0, 18), (4, 1, 12), (8, 1, 9), (9, 2, 10), (9, 1, 10),
             (9, 1, 7), (16, 0, 3), (16, 1, 3), (19, 0, 2), (20, 0, 2), (20, 1, 5)]
    traj2 = [(16, 1, 3), (17, 1, 3), (19, 2, 3)]
    traj3 = [(19, 0, 4), (19, 1, 5)]
    traj_total = heapq.merge((RunTimeTrajectorySeg(tid, *info) for tid, info in enumerate(traj1)),
                             (RunTimeTrajectorySeg(tid, *info) for tid, info in enumerate(traj2, 1)),
                             (RunTimeTrajectorySeg(tid, *info) for tid, info in enumerate(traj3, 6)))

    ans = [(6, 16, 1), (8, 19, 3), (9, 20, 1), (11, 21, 6)]
    searcher = SequentialSearcher((3, 21), lambda am, end, cond: [(len(am), end, len(cond))])
    query = list(searcher.search(traj_total))
    assert ans == query
