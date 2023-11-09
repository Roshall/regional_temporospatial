from region import GridRegion, TrajectoryClusteringRegion, Out3DRegion
from trajectory import Traj_Meta
from test_helper import *


class TestGrid:
    grid = GridRegion()
    for j in range(len(border[0])-1):
        for i in range(len(border[1])-1):
            grid.territory[i, j] = (border[0][j], border[1][i])

    def test_where_contain(self):
        Q = [(8, 16), (31, 63), (8, 63), (31, 16), (18, 37), (23, 49), (25, 71)]
        A = [(8, 16), (30, 60), (8, 60), (30, 16), (18, 36), (22, 48), (24, 68)]
        for q, a in zip(Q, A):
            assert self.grid.where_contain(q) == a

    def test_where_intersect(self):
        Q = [(8, 31, 16, 70), [8, 30, 16, 68], (11, 29, 17, 67)]
        A = [self.grid.territory, self.grid.territory, self.grid.territory[:-1, 1:-1]]
        for q, a in zip(Q, A):
            assert (self.grid.where_intersect(q) == a.flatten()).all()


class Test3DRegion:
    index = Out3DRegion(zip(keys, objs))

    def test_where_contain(self):
        for key, entry in zip(keys, objs):
            assert self.index.where_contain(key) is entry

    def test_where_intersect(self):
        res = list(self.index.where_intersect([10, 100, 20, 30, 16, 70]))
        assert len(res) == 4

    def test_add(self):
        objs = [list(range(5)), {i: str(i) for i in range(4)}]
        keys = [Traj_Meta(102, (19, 47)), Traj_Meta(5, (17, 55))]
        for key, entry in zip(keys, objs):
            self.index.add(key, entry)
            assert self.index.where_contain(key) is entry
