import heapq
import math
from collections import defaultdict
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
    user_idx = UserIdx()

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
        for pid, coord in enumerate(track_iter, 1):
            if (last_reg := reg.where_contain(coord)) is not first_reg:
                # 2. insert to index
                ts_beg = start + beg
                # FIXME: consider label
                user_idx.add(((track[start], track_life), (end - start, ts_beg)),
                             NaiveTrajectorySeg(tid, ts_beg, label, track[start:end]))
                start = end
                first_reg = last_reg
            end += 1

        ts_beg = start + beg
        user_idx.add(((track[start], track_life), (end - start, ts_beg)),
                     NaiveTrajectorySeg(tid, ts_beg, label, track[start:end]))
    return user_idx


def candidate_verified_queue(region, candidates, interval, buffer_size: int = 1024):
    cand_queue = heapq.merge(*candidates)
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

    res_seg = [RunTimeTrajectorySeg(segment.id, segment.begin + pos + 1,segment.label, len_)
               for pos, len_ in zip(break_pos, seg_lens) if len_ > 0]
    return res_seg


def yield_co_move(active_space, timestamp, interval, labels, cur_id):
    result_bag = defaultdict(lambda: 0)
    for tra_id, label in active_space:
        if tra_id <= timestamp:
            result_bag[tra_id] += 1
        elif timestamp - active_space[tra_id] < interval:
            break

    if result_bag.keys() == labels.keys():
        for tra_id in labels.keys():
            if result_bag[tra_id] < labels[tra_id]:
                return None

        return list(result_bag.keys()), active_space[cur_id], timestamp


def search(tempo_spat_idx, region: Box2D, labels, duration_range, interval):
    # FIXME: only rectangle region
    # TODO: ADD labels
    candidates, probation = tempo_spat_idx.perhaps_intersect(((region.bbox, duration_range), interval))
    traj_queue = heapq.merge(*probation, candidate_verified_queue(region, candidates, interval), key=attrgetter('begin'))
    end_queue = []
    next_end = math.inf
    active_space = {}
    pre_insert = {}
    new_ = -1

    for traj in traj_queue:
        begin = traj.begin
        if new_ == begin:
            assert traj.id in pre_insert
            pre_insert[traj.id] = traj
        else:
            new_ = begin
            if next_end == new_:
                while next_end == end_queue[0][0]:
                    end, tra_id = heapq.heappop(end_queue)
                    if tra_id in pre_insert:
                        del pre_insert[tra_id]
                    elif end - active_space[tra_id] < interval:
                        del active_space[tra_id]
                    else:
                        # produce result
                        res = yield_co_move(active_space, end, interval, labels, tra_id)
                        if res:
                            yield res
            for tra_id, tra in pre_insert.items():
                assert tra_id in active_space
                active_space[tra_id] = (tra.begin, tra.label)
                heapq.heappush(end_queue, (tra.len + tra.begin, tra_id))
            pre_insert.clear()
            next_end = end_queue[0][0]

    while end_queue:
        end, tra_id = heapq.heappop(end_queue)
        if end - active_space[tra_id] < interval:
            del active_space[tra_id]
        else:
            res = yield_co_move(active_space, end, interval, labels, tra_id)
            if res:
                yield res
