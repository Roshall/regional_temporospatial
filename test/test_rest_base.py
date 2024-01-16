from search.base import absorb, base_sliding
from search.verifier import obj_verify
from utilities.trajectory import Trajectory, TrajectoryIntervalSeg


def test_absorb():
    segs = [[9, 20, 1, 5, 29, 30, 6, 9, 20, 23, 27, 28, 26, 27, 28, 29],
            [5, 6, 3, 5, 8, 13]]
    ans = [[1, 5], [6, 23], [26, 30], [8, 13]]
    trajs = {i: Trajectory(i, 0, seg) for i, seg in enumerate(segs)}
    res = [[traj.begin, traj.begin + traj.len] for traj in absorb(trajs, 4)]
    assert res == ans


def test_base_sliding():
    traj_len = [[1, 21], [1, 5], [2, 10], [5, 16], [15, 20], [15, 21]]
    trajs = [TrajectoryIntervalSeg(i, itv[0], i & 1, itv[1]-itv[0]) for i, itv in enumerate(traj_len)]
    ans = [((0,1), [1, 4]), ((0,2,3), [5,8]),
           ((0,2,3), [6,9]), ((0, 3), [12, 15]), ((0, 4, 5), [15, 18]), ((0, 4, 5), [17, 20])]
    interval = 0, 20
    duration = 4
    labels = {0: 1, 1:1}
    from functools import partial
    label_verifier = partial(obj_verify, labels)
    res = [(tuple(pat.labels), pat.interval) for pat in base_sliding(trajs, interval, duration, label_verifier)]
    assert res == ans
