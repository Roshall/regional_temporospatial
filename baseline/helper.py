from collections import deque
from collections.abc import Iterable
from itertools import islice

from baseline.co_moving import CoMovementPattern


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


def obj_verify(target, label_map):
    if len(target) != len(label_map):
        return False
    for label in target:
        if target[label] > label_map[label]:
            return False
    return True


def len_filter(num_m, length):
    return [obj for obj in num_m if num_m[obj] >= length]


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
