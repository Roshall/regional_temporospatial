import heapq
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, MutableMapping
from collections.abc import Mapping, Sequence
from itertools import takewhile, islice
from operator import attrgetter
from typing import Iterator

import numpy as np

from utilities.box2D import Box2D
from utilities.trajectory import TrajectorySequenceSeg, TrajectoryIntervalSeg, BasicTrajectorySeg


def candidate_verified_queue(candidates: Iterable, region: Box2D, duration: int) -> Iterator[TrajectoryIntervalSeg]:
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


def yield_co_move(duration: int, labels: Mapping[int, int], active_space: MutableMapping[int, BasicTrajectorySeg],
                  timestamp: int, traj_required: Sequence) -> Iterable[tuple[list[int], int, int]]:
    """
    a co-movement checker respecting objects' label and count and their co-moving duration .
    Note that this function may modify `active_space`.
    :param duration: objects co-moving duration
    :param labels: {obj_label: count}
    :param active_space: {obj_id: trajectory}
    :param timestamp: current processing time
    :param traj_required: trajectories need processing
    :return: iterator of tuple(ids, start, end)
    """
    id_required = set(traj.id for traj in traj_required if timestamp - traj.begin >= duration)
    traj_cand = [active_space[traj_id] for traj_id in
                 takewhile(lambda x: timestamp - active_space[x].begin >= duration, active_space)]
    traj_cand.reverse()  # start at the least duration
    for traj in traj_required:
        del active_space[traj.id]

    result_bag = Counter(map(attrgetter('label'), traj_cand))
    ids = [traj.id for traj in traj_cand]

    # It's impossible for result bag to have more label types. because we filtered labels first.
    assert len(result_bag) <= len(labels)
    if len(result_bag) == len(labels):
        for tra_label in labels:
            if result_bag[tra_label] < labels[tra_label]:
                return
        res_pos = np.diff(np.fromiter(map(attrgetter('begin'), traj_cand),
                                      np.int32, len(traj_cand)), prepend=-1).nonzero()[0]
        timestamp -= 1  # the end point is exclusive
        for i in res_pos:
            traj = traj_cand[i]
            yield ids[i:], traj.begin, timestamp
            label = traj.label
            result_bag[traj.label] -= 1
            id_required.discard(traj.id)
            if not id_required or result_bag[label] < labels[label]:
                break


def group_until(queue, ts):
    if not queue or queue[0][0] > ts:
        return
    else:
        t, tid = heapq.heappop(queue)
        group = [tid]
        while queue:
            end = queue[0][0]
            if end == t:
                group.append(heapq.heappop(queue)[1])
            else:
                yield t, group
                if end > ts:
                    return
                t = end
                group = [heapq.heappop(queue)[1]]
        yield t, group
