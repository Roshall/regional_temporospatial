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
    # Note that prev[0] ⊃ prev[1] ⊃ prev[2] ⊃ ...
    prev = []
    for cur in sliding_result:
        new = []
        count = 0
        prev_end = prev[0].interval[1] if prev else None
        end = cur.interval[1]
        absort = True
        for pat in prev:
            new_pattern = pat & cur
            if len(new_pattern) == len(pat):  # pat is a subset of cur
                if len(new_pattern) == len(cur):  # pat equals to cur
                    new.append(pat.update_end(end))
                    count += 1
                else:
                    new.append(cur)
                break

            if len(new_pattern) == len(cur):  # cur is a proper subset of pat
                new.append(pat.update_end(end))
                count += 1
                continue

            if obj_varifier(new_pattern.label_count()):
                new_pattern.interval = [pat.interval[0], end]
                yield pat.update_end(prev_end)
                new.append(cur)
                cur = new_pattern
                count += 1
            else:
                new.append(cur)
                absort = False
                break
        else:
            new.append(cur)

        if absort:
            new.extend(islice(prev, count, None))
        else:
            for pat in islice(prev, count, None):
                yield pat.update_end(prev_end)
        prev = new

    if prev:
        end = prev[0].interval[1]
        for pat in prev:
            yield pat.update_end(end)
