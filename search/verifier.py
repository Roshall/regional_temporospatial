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
    verified = []
    for cand_seg in candidates:
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
    :return: sorted parts of the segment by `begin`simplify verified queue
    """
    mask = region.enclose(segment.points)
    in_pos = np.flatnonzero(mask)
    res = []
    if len(in_pos) != 0:
        sid, begin, label = segment.id, segment.begin, segment.label
        if mask.all():
            yield TrajectoryIntervalSeg(sid, begin, label, len(mask))
            return

        start_pos = np.flatnonzero(np.diff(in_pos, prepend=-2) > 1)
        seg_lens = np.diff(start_pos, append=len(in_pos))
        res_mask = np.flatnonzero(seg_lens >= duration)
        if mask[0] and seg_lens[0] < duration:
            res.append(TrajectoryIntervalSeg(sid, begin, label, seg_lens[0]))

        res.extend(TrajectoryIntervalSeg(sid, begin + in_pos[start_pos[m]], label, seg_lens[m]) for m in res_mask)

        if mask[-1] and seg_lens[-1] < duration:
            res.append(TrajectoryIntervalSeg(sid, begin + in_pos[start_pos[-1]], label, seg_lens[-1]))
    return res


def obj_verify(target, label_map):
    if len(target) != len(label_map):
        return False
    for label in target:
        if target[label] > label_map[label]:
            return False
    return True


def len_filter(num_m, length):
    return [obj for obj in num_m if num_m[obj] >= length]
