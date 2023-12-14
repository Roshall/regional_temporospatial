import heapq
import math
from collections import defaultdict, Counter
from collections.abc import Iterable, MutableMapping, Callable
from collections.abc import Mapping, Sequence
from functools import partial
from itertools import chain
from operator import attrgetter

import numpy as np

from indices import UserIdx
from indices.region import GridRegion
from utilities.box2D import Box2D
from utilities.trajectory import NaiveTrajectorySeg, RunTimeTrajectorySeg


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


def candidate_verified_queue(region, candidates, interval, buffer_size: int = 1024):
    cand_queue = heapq.merge(*chain.from_iterable(candidates))
    verified_heap = []
    # build queue
    for cand_seg in cand_queue:
        verified_heap.extend(verify_seg(region, cand_seg, interval))
        if len(verified_heap) >= buffer_size:
            break
    heapq.heapify(verified_heap)

    for cand_seg in cand_queue:
        verified_seg = verify_seg(region, cand_seg, interval)
        for seg in verified_seg:
            yield heapq.heapreplace(verified_heap, seg)

    while verified_heap:
        yield heapq.heappop(verified_heap)


def verify_seg(region, segment: NaiveTrajectorySeg, interval):
    mask = region.enclose(segment.points)
    break_pos = np.append(-1, np.where(mask == False))
    seg_lens = np.diff(break_pos, append=len(mask)) - 1
    # if the seg is cut into pieces, each is treated separately
    seg_lens_mid = seg_lens[1:-1]
    seg_lens_mid[seg_lens_mid < interval] = 0

    res_seg = [RunTimeTrajectorySeg(segment.id, segment.begin + pos + 1, segment.label, len_)
               for pos, len_ in zip(break_pos, seg_lens) if len_ > 0]
    return res_seg


def yield_co_move(duration: int, labels: Mapping[int, int], active_space: MutableMapping[int, tuple[int, int]],
                  timestamp: int, id_cand: Sequence):
    """
    a co-movement checker respecting objects' label and count and their co-moving duration .
    :param duration: objects co-moving duration
    :param labels: {obj_label: count}
    :param active_space: {obj_id: (begin, label)}
    :param timestamp: current processing time
    :param id_cand: all objects need processing
    :return: list of tuple(ids, start, end)
    """
    end_cand = []
    begin_cand = []
    label_cand = []
    for tra_id in id_cand:
        tinfo = active_space.pop(tra_id)
        if timestamp - tinfo[0] >= duration:
            end_cand.append(tra_id)
            begin_cand.append(tinfo[0])
            label_cand.append(tinfo[1])

    result_bag = Counter(label_cand)
    ids = id_cand[:]
    for traj_id, traj in active_space.items():
        if timestamp - traj[0] < duration:
            break
        else:
            result_bag[traj[1]] += 1
            ids.append(traj_id)

    # It's impossible for result bag to have more keys. because we filtered labels first.
    assert len(result_bag) <= len(labels)
    if len(result_bag) == len(labels):
        for tra_id in labels:
            if result_bag[tra_id] < labels[tra_id]:
                return []
        return [(ids, beg, timestamp) for beg in begin_cand]
    else:
        return []


def sequential_search(trajs: Iterable, interval, co_move_verifier: Callable[[MutableMapping, int, Sequence], Iterable]):
    start, finish = interval
    active_space = {}
    pre_insert = {}
    end_queue = []
    end_queue_push = partial(heapq.heappush, end_queue)
    end_queue_pop = partial(heapq.heappop, end_queue)

    new_ = start
    next_end = finish
    id_cand = []

    for traj in trajs:
        begin = max(traj.begin, start)
        if new_ == begin:
            assert traj.id not in pre_insert
            pre_insert[traj.id] = traj
        else:
            new_ = begin
            if next_end < new_:
                while end_queue and next_end == end_queue[0][0]:
                    _, tid = end_queue_pop()
                    if (revising := pre_insert.pop(tid, None)) is None:
                        id_cand.append(tid)
                    else:
                        end_queue_push((revising.len + revising.begin, tid))
                # produce result
                if id_cand:
                    yield from co_move_verifier(active_space, next_end, id_cand)
                    id_cand.clear()

            next_end = update(active_space, end_queue, end_queue_push, pre_insert)
            pre_insert.clear()
            pre_insert[traj.id] = traj

    next_end = update(active_space, end_queue, end_queue_push, pre_insert)

    while end_queue:
        if next_end >= finish:
            yield from co_move_verifier(active_space, finish, [info[1] for info in end_queue])
            break
        while end_queue and next_end == end_queue[0][0]:
            id_cand.append(end_queue_pop()[1])
        yield from co_move_verifier(active_space, next_end, id_cand)
        id_cand.clear()
        next_end = end_queue[0][0]


def update(active_space, end_queue, end_queue_push, pre_insert):
    for tra_id, tra in pre_insert.items():
        assert tra_id not in active_space
        active_space[tra_id] = (tra.begin, tra.label)
        end_queue_push((tra.len + tra.begin, tra_id))
    next_end = end_queue[0][0]
    return next_end


def base_query(tempo_spat_idx, region: Box2D, labels: Mapping, duration_range, interval: int):
    # FIXME: only rectangle region
    candidates, probation = zip(*(tempo_spat_idx[label].perhaps_intersect(((region.bbox, duration_range), interval))
                                  for label in labels))
    # candidates, probation = tempo_spat_idx.perhaps_intersect(((region.bbox, duration_range), interval))
    traj_queue = heapq.merge(*chain.from_iterable(probation), candidate_verified_queue(region, candidates, interval),
                             key=attrgetter('begin'))
    verifier = partial(yield_co_move, duration_range[0], labels)
    sequential_search(traj_queue, interval, verifier)
