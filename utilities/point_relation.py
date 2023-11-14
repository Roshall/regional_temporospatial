import numpy as np


def enclose_tester(reference_points):
    """
    tested region represent by three points forming two perpendicular lines:
    p[1](A)
    |
    |
    p[0](B)____p[2](C)
    :param reference_points: points array.
    :return: boolean array
    """
    points = np.asarray(reference_points)
    reference = np.empty((4, len(reference_points[0])))
    vectors = points[1:] - points[0]  # [BA, BC].T
    reference[:2] = vectors
    reference[2] = np.diag(vectors @ vectors.T)  # [BA@BA, BC@BC].T
    reference[3] = points[0]

    def enclose(tested_points):
        test_vectors = np.asarray(tested_points) - reference[-1]  # BP
        stats = test_vectors @ reference[:2].T  # [BP@BA, BP@BC]
        mask1 = stats >= 0
        mask2 = stats <= reference[2]  # [BP@BA < BA@BA, BP@BC < BC@BC]
        return (mask1 & mask2).all(axis=1)
    return enclose
