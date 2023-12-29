import numpy as np


class Box2D:
    def __init__(self, bbox):
        """
        build box by bbox
        :param bbox: (dim0_min, dim0_max, dim1_min, dim1_max)
        """
        self.bbox = bbox
        self._test_meta = None
        self._meta()

    def _point_rep(self):
        """
        transform bbox into 3 points to represent a box
        p[1](A)
        |
        |
        p[0](B)____p[2](C)
        :return: three points
        """
        points = np.empty((3, 2))
        points[0] = self.bbox[::2]
        points[1] = self.bbox[::3]
        points[2] = self.bbox[1:3]
        return points

    def _meta(self):
        """
        build useful data for testing if a point within this box
        :return: None
        """
        points_rep = self._point_rep()
        reference = np.empty((4, len(points_rep[0])))
        vectors = points_rep[1:] - points_rep[0]  # [BA, BC].T
        reference[:2] = vectors.T
        reference[2] = np.diag(vectors @ vectors.T)  # [BA@BA, BC@BC]
        reference[3] = points_rep[0]
        self._test_meta = reference

    def enclose(self, points):
        """
        test whether points are in this box
        :param points: 2d points, iterable
        :return: boolean array
        """
        test_vectors = points - self._test_meta[-1]  # BP
        stats = test_vectors @ self._test_meta[:2]  # [BP@BA.T, BP@BC.T]
        mask1 = stats >= 0
        mask2 = stats <= self._test_meta[2]  # [BP@BA < BA@BA, BP@BC < BC@BC]
        return (mask1 & mask2).all(axis=1)
