from bisect import bisect_right
import numpy as np

from .index_interface import Index
from utilities.config import config


class GridRegion(Index):
    def __init__(self):
        self.borders = config.gird_border
        # this marker marks the starting point of each grid, while $borders
        # contains the end corners, thus removing the last element
        self.terr_marker = [dim[:-1] for dim in self.borders]
        shape = tuple(map(len, self.terr_marker))
        self.territory = np.empty(shape[::-1], dtype=object)  # x: column, y: row etc.

    def add(self, loc, entry):
        self.territory[self._point2idx(loc)] = entry

    def _point2idx(self, loc):
        return tuple(
            reversed([bisect_right(dim_grid, dim_point) - 1 for dim_grid, dim_point in zip(self.terr_marker, loc)]))

    def _include(self, loc):
        for dim_grid, dim_point in zip(self.borders, loc):
            if dim_point < dim_grid[0] or dim_point > dim_grid[-1]:
                return False
        return True

    def where_contain(self, loc):
        if not self._include(loc):
            return
        return self.territory[self._point2idx(loc)]

    def where_intersect(self, bbox):
        bbox_iter = iter(bbox)
        region_ls = []
        for dim_grid in self.borders:
            dmin = bisect_right(dim_grid, next(bbox_iter)) - 1
            dmax = bisect_right(dim_grid, next(bbox_iter), dmin)  # the end of a slice is exclusive
            region_ls.append(slice(dmin, dmax))
        res = self.territory[tuple(reversed(region_ls))]
        return res[res.nonzero()]


class Out3DRegion(Index):
    def __init__(self, data=None):
        import btree
        self.duration_index = btree.BtreeMap()
        if data is not None:
            for key, entry in data:
                self.add(key, entry)

    def where_contain(self, key):
        return (reg := self.duration_index.where_contain(key.duration)) and reg.where_contain(key.loc)

    def where_intersect(self, bbox):
        for reg in self.duration_index.where_intersect(bbox[:2]):
            yield from reg.where_intersect(bbox[2:])

    def add(self, key, entry) -> None:
        if (reg := self.duration_index.where_contain(key.duration)) is None:
            reg = GridRegion()
            self.duration_index.add(key.duration, reg)
        reg.add(key.loc, entry)


class TrajectoryClusteringRegion(Index):
    def add(self, key, entry) -> None:
        pass

    def __init__(self, data):
        pass

    def where_contain(self, loc):
        pass

    def where_intersect(self, bbox):
        pass
