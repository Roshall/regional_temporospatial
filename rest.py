import heapq
from collections import defaultdict, Counter
from collections.abc import Iterable, MutableMapping
from collections.abc import Mapping, Sequence
from copy import copy
from functools import partial
from itertools import chain, takewhile
from operator import attrgetter

import numpy as np

from indices import UserIdx
from indices.region import GridRegion
from utilities.box2D import Box2D
from utilities.trajectory import NaiveTrajectorySeg, RunTimeTrajectorySeg, BasicTrajectorySeg


def build_tempo_spatial_index(trajs):
    # 1. find trajectory's region
    # config.gird_border = gen_border(trajs.bbox, 10, 15)
    reg = GridRegion()
    user_idx = defaultdict(UserIdx)

    # we need to fill the territory with distinct objects
    class _C:
        def __init__(self, arg):
            pass

    fill = np.vectorize(_C)
    reg.territory = fill(reg.territory)

    for tid, label, beg, track in trajs.traj_track:
        track_iter = iter(track)
        first_reg = reg.where_contain(next(track_iter))
        track_life = len(track)
        start, end = 0, 1
        # if some point jumps to another region, we cut the trajectory to a new segment.
        for coord in track_iter:
            if (last_reg := reg.where_contain(coord)) is not first_reg:
                # 2. insert to index
                ts_beg = start + beg
                user_idx[label].add(((track[start], track_life), (end - start, ts_beg)),
                                    NaiveTrajectorySeg(tid, ts_beg, label, track[start:end]))
                start = end
                first_reg = last_reg
            end += 1

        ts_beg = start + beg
        user_idx[label].add(((track[start], track_life), (end - start, ts_beg)),
                            NaiveTrajectorySeg(tid, ts_beg, label, track[start:end]))
    return user_idx


def candidate_verified_queue(region, candidates, duration, buffer_size: int = 1024):
    cand_queue = heapq.merge(*chain.from_iterable(candidates))
    verified_heap = []
    # build queue
    for cand_seg in cand_queue:
        verified_heap.extend(verify_seg(region, cand_seg, duration))
        if len(verified_heap) >= buffer_size:
            break
    heapq.heapify(verified_heap)

    for cand_seg in cand_queue:
        verified_seg = verify_seg(region, cand_seg, duration)
        for seg in verified_seg:
            yield heapq.heapreplace(verified_heap, seg)

    while verified_heap:
        yield heapq.heappop(verified_heap)


def verify_seg(region, segment: NaiveTrajectorySeg, duration):
    mask = region.enclose(segment.points)
    break_pos = np.append(-1, np.where(mask == False))
    seg_lens = np.diff(break_pos, append=len(mask)) - 1
    # if the seg is cut into pieces, each is treated separately
    seg_lens_mid = seg_lens[1:-1]
    seg_lens_mid[seg_lens_mid < duration] = 0

    res_seg = [RunTimeTrajectorySeg(segment.id, segment.begin + pos + 1, segment.label, len_)
               for pos, len_ in zip(break_pos, seg_lens) if len_ > 0]
    return res_seg


def yield_co_move(duration: int, labels: Mapping[int, int], active_space: MutableMapping[int, BasicTrajectorySeg],
                  timestamp: int, traj_required: Sequence):
    """
    a co-movement checker respecting objects' label and count and their co-moving duration .
    Note that this function may modify active_space.
    :param duration: objects co-moving duration
    :param labels: {obj_label: count}
    :param active_space: {obj_id: trajectory}
    :param timestamp: current processing time
    :param traj_required: trajectories need processing
    :return: iterator of tuple(ids, start, end)
    """
    id_required = set(traj.id for traj in traj_required if timestamp - traj.begin >= duration)
    traj_cand = [active_space[traj_id] for traj_id in takewhile(lambda x: timestamp - active_space[x].begin >= duration, active_space)]
    traj_cand.reverse()   # start at the least duration
    for traj in traj_required:
        del active_space[traj.id]

    result_bag = Counter(map(attrgetter('label'), traj_cand))
    ids = [traj.id for traj in traj_cand]

    # It's impossible for result bag to have more keys. because we filtered labels first.
    assert len(result_bag) <= len(labels)
    if len(result_bag) == len(labels):
        for tra_label in labels:
            if result_bag[tra_label] < labels[tra_label]:
                return
        res_pos = np.diff(np.fromiter(map(attrgetter('begin'), traj_cand), np.int32), prepend=-1).nonzero()[0]
        timestamp -= 1  # the end point is exclusive
        for i in res_pos:
            traj = traj_cand[i]
            yield ids[i:], traj.begin, timestamp
            label = traj.label
            result_bag[traj.label] -= 1
            id_required.discard(traj.id)
            if not id_required or result_bag[label] < labels[label]:
                break


def group_traj_by_time(trajectories, head=None):
    if head is None:
        if (head := next(trajectories, None)) is None:
            return
    t = head.begin
    pre = {head.id: head}
    for traj in trajectories:
        if traj.begin == t:
            pre[traj.id] = traj
        else:
            yield t, pre
            t = traj.begin
            pre = {traj.id: traj}
    if pre:
        yield t, pre


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


def base_query(tempo_spat_idx, region: Box2D, labels: Mapping, duration_range, interval):
    # FIXME: only rectangle region
    candidates, probation = zip(*(tempo_spat_idx[label].where_intersect(((region.bbox, duration_range), interval))
                                  for label in labels))
    # candidates, probation = tempo_spat_idx.perhaps_intersect(((region.bbox, duration_range), interval))
    traj_queue = heapq.merge(*chain.from_iterable(probation),
                             candidate_verified_queue(region, candidates, duration_range[0]), key=attrgetter('begin'))
    verifier = partial(yield_co_move, duration_range[0], labels)
    searcher = SequentialSearcher(interval, verifier)
    return searcher.search(traj_queue)


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

    def _head(self, trajs):
        start = self.interval[0]
        for traj in trajs:  # gather trajs on the starting border
            if traj.begin <= start:
                self.etq_push((traj.begin + traj.len, traj.id))
                (traj := copy(traj)).begin = start
                self.playground[traj.id] = traj
            else:
                return traj

    def search(self, trajs: Iterable[RunTimeTrajectorySeg | NaiveTrajectorySeg]):
        begin, finish = self.interval

        head = self._head(trajs)
        if self.etq:
            next_end = self.etq[0][0]
        elif head:
            next_end = head.begin + head.len
        else:
            return

        if head:
            for t, pre_insert in group_traj_by_time(trajs, head):
                if next_end <= t:
                    yield from self._yield_until(t, pre_insert)
                next_end = self._update(pre_insert)

        if next_end < finish:
            yield from self._yield_until(finish - 1, {})
        yield from self.verify(finish, [self.playground[info[1]]for info in self.etq])
