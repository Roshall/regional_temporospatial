from _bisect import bisect_right
from itertools import islice
from typing import Iterable

import numpy as np

from utilities.box2D import Box2D
from utilities.trajectory import TrajectoryIntervalSeg, TrajectorySequenceSeg


def candidate_verified_queue(candidates: Iterable, region: Box2D, duration: int) -> Iterable[TrajectoryIntervalSeg]:
    """
    verify trajectories and find the segments within region
    :param region: a box object with `enclose` function implemented
    :param candidates: trajectories source
    :param duration: the least lifetime of a segment
    :return: trajectory segments within region, which are guaranteed to be sorted by `begin` if `candidates` is sorted.
    """
    cand_queue = iter(candidates)
    if (first := next(cand_queue, None)) is None:
        return
    verified = verify_seg(first, region, duration)
    for cand_seg in cand_queue:
        segs = verify_seg(cand_seg, region, duration)
        if segs:
            pos = bisect_right(verified, segs[0])
            yield from islice(verified, pos)
            if pos != len(verified):
                segs.extend(islice(verified, pos, None))
                segs.sort()
            verified = segs
    yield from verified


def verify_seg(segment: TrajectorySequenceSeg, region: Box2D, duration: int) -> list[TrajectoryIntervalSeg]:
    """
    verify a trajectory segment, and find parts within region
    :param segment: trajectory sequence segment.
    :param region: a box object with `enclose` function implemented
    :param duration: the least lifetime of a segment
    :return: sorted parts of the segment by `begin`
    """
    mask = region.enclose(segment.points)
    break_pos = np.append(-1, np.where(mask == False))
    seg_lens = np.diff(break_pos, append=len(mask)) - 1
    if len(break_pos) > 1:  # if the seg is cut into pieces, each is treated separately
        seg_lens_mid = seg_lens[1:-1]
        seg_lens_mid[seg_lens_mid < duration] = 0

    return [TrajectoryIntervalSeg(segment.id, segment.begin + pos + 1, segment.label, len_)
            for pos, len_ in zip(break_pos, seg_lens) if len_ > 0]


def obj_verify(target, label_map):
    if len(target) != len(label_map):
        return False
    for label in target:
        if target[label] > label_map[label]:
            return False
    return True


def len_filter(num_m, length):
    return [obj for obj in num_m if num_m[obj] >= length]
