from search.base import absorb
from utilities.trajectory import Trajectory


def test_absorb():
    segs = [[9, 20, 1, 5, 29, 30, 6, 9, 20, 23, 27, 28, 26, 27, 28, 29],
            [5, 6, 3, 5, 8, 13]]
    ans = [[1, 5], [6, 23], [26, 30], [8, 13]]
    trajs = {i: Trajectory(i, 0, seg) for i, seg in enumerate(segs)}
    res = [[traj.begin, traj.begin + traj.len] for traj in absorb(trajs, 4)]
    assert res == ans
