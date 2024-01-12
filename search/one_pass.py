import heapq
from collections.abc import Mapping, Iterable
from copy import copy
from functools import partial
from itertools import chain, groupby
from operator import attrgetter

from search.rest import yield_co_move, group_until
from search.verifier import candidate_verified_queue
from utilities.box2D import Box2D
from utilities.trajectory import TrajectoryIntervalSeg, TrajectorySequenceSeg


class SequentialSearcher:
    def __init__(self, interval, verifier):
        self.interval = interval
        self.playground = {}
        self.verify = partial(verifier, self.playground)
        end_time_queue = []
        self.etq_push = partial(heapq.heappush, end_time_queue)
        self.etq = end_time_queue

    def _yield_until(self, ts, pre_insert):
        for t, group in group_until(self.etq, ts):
            if t < ts:
                yield from self.verify(t, [self.playground[tid] for tid in group])
            elif t == ts:
                t_candi = []
                for tid in group:
                    if (revising := pre_insert.pop(tid, None)) is None:
                        t_candi.append(self.playground[tid])
                    else:
                        self.etq_push((revising.len + revising.begin, tid))
                if t_candi:
                    yield from self.verify(ts, t_candi)
                return

    def _update(self, pre_insert):
        for tra_id, tra in pre_insert.items():
            assert tra_id not in self.playground
            self.playground[tra_id] = tra
            self.etq_push((tra.len + tra.begin, tra_id))
        return self.etq[0][0]

    def _head(self, grouped_trajs):
        start = self.interval[0]
        for ts, trajs in grouped_trajs:  # gather trajs on the starting border
            for traj in trajs:
                self.etq_push((traj.begin + traj.len, traj.id))
                if ts < start:
                    (traj := copy(traj)).begin = start
                self.playground[traj.id] = traj
            if ts > start:
                break

    def search(self, trajs: Iterable[TrajectoryIntervalSeg | TrajectorySequenceSeg]):
        begin, finish = self.interval
        ts_grouped_traj = groupby(trajs, key=attrgetter('begin'))
        self._head(ts_grouped_traj)
        if self.etq:
            next_end = self.etq[0][0]
        else:
            return

        for t, group in ts_grouped_traj:
            pre_insert = {traj.id: traj for traj in group}
            if next_end <= t:
                yield from self._yield_until(t, pre_insert)
            next_end = self._update(pre_insert)

        if next_end < finish + 2:  # in a segment, end point is exclusive
            yield from self._yield_until(finish + 1, {})
        yield from self.verify(finish, [self.playground[info[1]] for info in self.etq])


def one_pass_search(tempo_spat_idx, region: Box2D, labels: Mapping, duration_range, interval):
    # FIXME: only rectangle region
    candidates, probation = zip(*(tempo_spat_idx[label].where_intersect(((region.bbox, duration_range), interval))
                                  for label in labels))
    traj_queue = heapq.merge(*chain.from_iterable(probation),
                             candidate_verified_queue(heapq.merge(*chain.from_iterable(candidates)),
                                                      region, duration_range[0]),
                             key=attrgetter('begin'))
    verifier = partial(yield_co_move, duration_range[0], labels)
    searcher = SequentialSearcher(interval, verifier)
    return searcher.search(traj_queue)
