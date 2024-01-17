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
            if len(remains) > 0:
                res = np.empty(remains.size + 2, dtype=np.int32)
                res[[0, -1]] = seq[[0, -1]]
                res[1:-1] = remains
                res = res.reshape(-1, 2)
                res[:, 1] -= res[:, 0]  # length
                yield from [TrajectoryIntervalSeg(tid, beg, traj.label, l)
                            for beg, l in res if l >= duration]
                continue

        s_len = traj.seg[1] - traj.seg[0]
        if s_len >= duration:
            yield TrajectoryIntervalSeg(tid, traj.seg[0], traj.label, s_len)


class BaseSliding:
    def __init__(self, trajectories: list[TrajectoryIntervalSeg], interval, dur, label_verifier):
        self.ts_grouped_traj = trajectories
        self.label_verifier = label_verifier
        self.interval = interval
        self.dur = dur
        self.end_q = []
        self.label_m = {}
        self.eq_push = partial(heapq.heappush, self.end_q)
        self.eq_group_pop = partial(group_until, self.end_q)

    def __iter__(self):
        terminal = self.interval[1]
        yield from self._init_state()

        for ts, trajs in self.ts_grouped_traj:
            if (end := ts + self.dur - 1) >= terminal:
                if end == terminal:
                    self._add(trajs)
                yield from self.check_new_pattern([terminal - self.dur + 1, terminal])
                break

            if end > self.end_q[0][0]:
                for ts_end, group in self.eq_group_pop(end):
                    yield from self.check_new_pattern([ts_end - self.dur + 1, ts_end])
                    self._remove(group)

            self._add(trajs)
            yield from self.check_new_pattern([ts, end])

            if end == self.end_q[0][0]:
                for _, group in self.eq_group_pop(end):  # there must be only one group
                    self._remove(group)
        else:
            if self.end_q:
                yield from self.check_new_pattern([terminal-self.dur+1, terminal])

    def _remove(self, group):
        for tid in group:
            del self.label_m[tid]

    def _init_state(self):
        trajectories = self.ts_grouped_traj
        trajectories.sort(key=attrgetter('begin'))
        ts_grouped_traj = groupby(trajectories, key=attrgetter('begin'))
        start = self.interval[0]
        for ts, trajs in ts_grouped_traj:
            if ts > start:
                break
            else:
                for tra in trajs:
                    if start + tra.len > self.dur:
                        self.label_m[tra.id] = tra.label
                        self.end_q.append((tra.len + tra.begin - 1, tra.id))

        if not self.end_q:
            self._add(trajs)
        else:
            yield from self.check_new_pattern([start, start+self.dur-1])
        self.ts_grouped_traj = chain([(ts, trajs)], ts_grouped_traj)

    def check_new_pattern(self, interval=None):
        if self.label_verifier(Counter(self.label_m.values())):
            yield CoMovementPattern(self.label_m.copy(), interval)

    def _add(self, trajs):
        for tra in trajs:
            self.label_m[tra.id] = tra.label
            self.eq_push((tra.len + tra.begin - 1, tra.id))


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
        partial_res = BaseSliding(trajs, interval, duration_range[0], label_verifier)
        return expend(partial_res, label_verifier)
    else:
        return iter([])


