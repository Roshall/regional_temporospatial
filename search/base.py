import heapq
from collections import Counter
from collections.abc import Mapping
from functools import partial
from itertools import chain, groupby
from operator import attrgetter

import numpy as np

from search.co_moving import CoMovementPattern
from search.rest import group_until, expend
from search.verifier import candidate_verified_queue, obj_verify
from utilities.box2D import Box2D
from utilities.trajectory import TrajectoryIntervalSeg, Trajectory


def absorb(trajectories: Mapping[int, Trajectory], duration):
    """
    merge segments from the same object into a single segment
    :param trajectories: a map: id -> trajectory
    :param duration: co-movement duration
    :return: a generator with trajectories herein.
    """
    for tid, traj in trajectories.items():
        seq = traj.seg
        if len(seq) > 2:
            seq = np.array(traj.seg)
            seq.sort()
            # When we cut a trajectory, the end point doesn't actually belong to the segment
            # so if two segments are adjacent, one's end == another begin.
            mid = seq[1:-1].reshape(-1, 2)
            remains = mid[np.flatnonzero(mid[:, 1] - mid[:, 0])].reshape(-1)
            if remains:
                res = np.empty(remains.size + 2)
                res[[0, -1]] = seq[[0, -1]]
                res[1:-1] = remains
                res = res.reshape(-1, 2)
                len_ = res[:, 1] - res[:, 0]
                yield from [TrajectoryIntervalSeg(tid, beg, traj.label, l) for beg, l in zip(res[:, 1], len_) if l >= duration]
                continue

        s_len = traj.seg[1] - traj.seg[0]
        if s_len >= duration:
            yield TrajectoryIntervalSeg(tid, traj.seg[0], traj.label, s_len)


def base_sliding(trajs, labels, interval, dur, label_verifier):

    trajs.sort(key=attrgetter('begin'))
    # sliding windows alike method
    ts_grouped_traj = groupby(trajs, key=attrgetter('begin'))

    start, end = interval
    end_q = []
    label_m = {}
    for ts, trajs in ts_grouped_traj:
        if ts > start:
            break
        else:
            for tra in trajs:
                if start + tra.len > dur:
                    label_m[tra.id] = tra.label
                    end_q.append((tra.len + tra.begin - 1, tra.id))
    heapq.heapify(end_q)
    if label_verifier(Counter(label_m.values())):
        yield CoMovementPattern(label_m.copy(), [start.copy(), start+dur-1])

    sure = True
    for ts, trajs in chain([(ts, trajs)], ts_grouped_traj):
        if (end := ts + dur - 1) > end_q[0][0]:
            for ts_end, group in group_until(end_q, end):
                if sure or label_verifier(Counter(label_m.values())):
                    yield CoMovementPattern(label_m.copy(), [ts_end - dur + 1, ts_end])
                for tid in group:
                    del label_m[tid]
                sure = False

        for tra in trajs:
            label_m[tra.id] = tra.label
            heapq.heappush(end_q, (tra.len + tra.begin - 1, tra.id))
        if sure or label_verifier(Counter(label_m.values())):
            yield CoMovementPattern(label_m.copy(), [ts, end])
            sure = True

        if end == end_q[0][0]:
            for _, group in group_until(end_q, end):  # there must be only one group
                for _, tid in group:
                    del label_m[tid]
            sure = False


def base_search(spat_tempo_idx, region: Box2D, labels: Mapping, duration_range, interval):
    label_verifier = partial(obj_verify, labels)
    # Note that spat_tempo should use fuzzy search but not fuzzy inner all
    candidates, probation = zip(*(spat_tempo_idx[label].where_intersect(((region.bbox, duration_range), interval))
                                  for label in labels))
    verified = candidate_verified_queue(candidates, region, duration_range)
    visited = {}
    for seg in chain([verified, probation]):
        if (old := visited.get(seg.id, None)) is None:
            visited[seg.id] = Trajectory(seg.id, seg.label, [seg.begin, seg.begin + seg.len])
        else:
            old.seg.extend([seg.begin, seg.begin + seg.len])

    trajs = list(absorb(visited, duration_range[0]))
    if trajs:
        partial_res = base_sliding(trajs, labels, interval, duration_range[0], label_verifier)
        return expend(partial_res, label_verifier)
    else:
        return iter([])


