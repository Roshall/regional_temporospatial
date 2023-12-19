import heapq
from collections import defaultdict, Counter
from collections.abc import Iterable, MutableMapping
from collections.abc import Mapping, Sequence
from copy import copy
from functools import partial
from itertools import chain
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
                  timestamp: int, traj_cand: Sequence):
    """
    a co-movement checker respecting objects' label and count and their co-moving duration .
    Note that this function may modify active_space.
    :param duration: objects co-moving duration
    :param labels: {obj_label: count}
    :param active_space: {obj_id: trajectory}
    :param timestamp: current processing time
    :param traj_cand: all trajectories need processing
    :return: iterator of tuple(ids, start, end)
    """
    for traj in traj_cand:
        del active_space[traj.id]
    traj_cand = [traj for traj in traj_cand if timestamp - traj.begin >= duration]
    traj_cand.sort(key=attrgetter('begin'), reverse=True)
    result_bag = Counter(map(attrgetter('label'), traj_cand))
    ids = [traj.id for traj in traj_cand]
    for traj_id, traj in active_space.items():
        if timestamp - traj.begin < duration:
            break
        else:
            result_bag[traj.label] += 1
            ids.append(traj_id)

    # It's impossible for result bag to have more keys. because we filtered labels first.
    assert len(result_bag) <= len(labels)
    if len(result_bag) == len(labels):
        for tra_label in labels:
            if result_bag[tra_label] < labels[tra_label]:
                return iter([])
        for i in range(len(traj_cand)):
            traj = traj_cand[i]
            yield ids[i:], traj.begin, timestamp
            label = traj.label
            result_bag[label] -= 1
            if result_bag[label] < labels[label]:
                break
    else:
        return iter([])


def sequential_search(trajs: Iterable[RunTimeTrajectorySeg | NaiveTrajectorySeg],
                      interval, co_move_verifier: Callable[[MutableMapping, int, Sequence], Iterable]):
    start, finish = interval
    active_space = {}
    pre_insert = {}
    end_queue = []
    end_queue_push = partial(heapq.heappush, end_queue)
    end_queue_pop = partial(heapq.heappop, end_queue)
    next_end = math.inf

    for traj in trajs:  # gather trajs on the starting border
        if traj.begin <= start:
            end_queue_push((traj.begin + traj.len, traj.id))
            (traj := copy(traj)).begin = start
            active_space[traj.id] = traj
        else:
            pre_insert[traj.id] = traj
            if end_queue:
                next_end = end_queue[0][0]
            new_ = traj.begin
            break
    else:  # no result
        return iter([])

    id_cand = []
    for traj in trajs:
        begin = traj.begin
        # gather all traj with the same beginning time and processing them all at a time.
        if new_ == begin:
            assert traj.id not in pre_insert
            pre_insert[traj.id] = traj
        else:
            new_ = begin
            if next_end < new_:
                yield from yield_at_time(next_end, active_space, co_move_verifier, end_queue, end_queue_pop,
                                         end_queue_push, id_cand, pre_insert)

            next_end = update(active_space, end_queue, end_queue_push, pre_insert)
            pre_insert.clear()
            pre_insert[traj.id] = traj
    yield from yield_at_time(next_end, active_space, co_move_verifier, end_queue, end_queue_pop, end_queue_push, id_cand,
                             pre_insert)
    if end_queue[0][0] == next_end:
        _, tid = end_queue_pop()
        if (revising := pre_insert.pop(tid, None)) is None:
            id_cand.append(active_space[tid])
            yield from co_move_verifier(active_space, next_end, id_cand)
        else:
            end_queue_push((revising.len + revising.begin, tid))
    if pre_insert:
        update(active_space, end_queue, end_queue_push, pre_insert)

    while end_queue:
        next_end = end_queue[0][0]
        if next_end >= finish:
            yield from co_move_verifier(active_space, finish, [active_space[info[1]] for info in end_queue])
            break
        while end_queue and next_end == end_queue[0][0]:
            id_cand.append(active_space[end_queue_pop()[1]])
        yield from co_move_verifier(active_space, next_end, id_cand)
        id_cand.clear()


def yield_at_time(next_end, active_space, co_move_verifier, end_queue, end_queue_pop, end_queue_push, id_cand,
                  pre_insert):
    while end_queue and next_end == end_queue[0][0]:  # process trajs ending at this time all at once
        _, tid = end_queue_pop()
        # If there is a traj ending and beginning at the same time, it passes through two regions
        # hence we remove this start point, and revising its end time
        if (revising := pre_insert.pop(tid, None)) is None:
            id_cand.append(active_space[tid])
        else:
            end_queue_push((revising.len + revising.begin, tid))
    # produce result
    if id_cand:
        res = co_move_verifier(active_space, next_end, id_cand)
        id_cand.clear()
        return res
    else:
        return iter([])


def update(active_space, end_queue, end_queue_push, pre_insert):
    for tra_id, tra in pre_insert.items():
        assert tra_id not in active_space
        active_space[tra_id] = tra
        end_queue_push((tra.len + tra.begin, tra_id))
    next_end = end_queue[0][0]
    return next_end


def base_query(tempo_spat_idx, region: Box2D, labels: Mapping, duration_range, interval):
    # FIXME: only rectangle region
    candidates, probation = zip(*(tempo_spat_idx[label].where_intersect(((region.bbox, duration_range), interval))
                                  for label in labels))
    # candidates, probation = tempo_spat_idx.perhaps_intersect(((region.bbox, duration_range), interval))
    traj_queue = heapq.merge(*chain.from_iterable(probation),
                             candidate_verified_queue(region, candidates, duration_range[0]), key=attrgetter('begin'))
    verifier = partial(yield_co_move, duration_range[0], labels)
    return sequential_search(traj_queue, interval, verifier)
