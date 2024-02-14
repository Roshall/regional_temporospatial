import heapq
from collections import Counter, deque
from collections.abc import Iterable, MutableMapping
from collections.abc import Mapping, Sequence
from itertools import takewhile, islice
from operator import attrgetter

import numpy as np

from search.co_moving import CoMovementPattern
from utilities.trajectory import BasicTrajectorySeg


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


def sliding_window(df, win_len):
    df_iter = iter(df)
    window = deque(islice(df_iter, win_len), maxlen=win_len)
    if not window:
        return
    else:
        yield window
    for frame in df_iter:
        window.append(frame)
        yield window


def expend(sliding_result: Iterable[CoMovementPattern], obj_varifier):
    # $prev stores seen patterns. Every pattern is a set of objs that co-moves a certain period
    # Note that in terms of object set, prev[0] ⊃ prev[1] ⊃ prev[2] ⊃ ...
    sliding_result = iter(sliding_result)
    cur = next(sliding_result, None)
    if cur is None:
        return
    prev = [cur]
    for cur in sliding_result:
        prev_end = prev[0].end
        if cur.start > prev_end + 1:  # not consecutive in time interval
            for p in prev:
                p.end = prev_end
                yield p
            prev = [cur]
            continue

        new = []
        count = 0
        cur_end = cur.end
        absort = True
        for pat in prev:
            new_pattern = pat & cur
            if len(new_pattern) == len(pat):  # pat is a subset of cur
                if len(new_pattern) == len(cur):  # pat equals to cur
                    pat.end = cur_end
                else:
                    new.append(cur)
                break

            elif len(new_pattern) == len(cur):  # cur is a proper subset of pat
                yield pat
                cur.start = pat.start
                count += 1

            elif obj_varifier(new_pattern.label_count()):  # really a new pattern
                new_pattern.interval = [pat.start, cur_end]
                pat.end = prev_end
                yield pat
                new.append(cur)
                cur = new_pattern
                count += 1
            else:
                new.append(cur)
                absort = False
                break
        else:
            new.append(cur)

        rests = islice(prev, count, None)
        if absort:
            new.extend(rests)
        else:
            for pat in rests:
                pat.end = prev_end
                yield pat
        prev = new

    if prev:
        cur_end = prev[0].end
        for pat in prev:
            pat.end = cur_end
            yield pat
