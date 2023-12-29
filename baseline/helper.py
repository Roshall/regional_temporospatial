from collections import deque
from itertools import islice


def sliding_window(df, win_len):
    window = deque(islice(df, win_len), maxlen=win_len)
    if not window:
        return
    else:
        yield window
    for frame in df:
        window.append(frame)
        yield window


def obj_verify(target, label_map):
    if len(target) != len(label_map):
        return False
    for label in target:
        if target[label] > label_map[label]:
            return False
    return True


def len_verify(num_m, length):
    for obj in num_m:
        if num_m[obj] < length:
            return False
    return True


def expend(sliding_result, obj_varifier):
    if (first := next(sliding_result, None)) is None:
        return
    win_len = sliding_result.win_len
    start, num_m = first
    length = win_len
    for low, principle in sliding_result:
        new_pattern = num_m & principle
        if obj_varifier(new_pattern):
            flag1, flag2 = new_pattern == num_m, new_pattern == principle
            if flag1 or flag2:
                if not flag2:
                    yield low, win_len, tuple(principle)
                elif not flag1:
                    yield start, length, tuple(num_m)

                length += win_len
            else:
                yield start, length, tuple(num_m),
                yield low, win_len, tuple(principle),
                start, length = low, win_len

            num_m = new_pattern
        else:
            yield start, start + len, tuple(num_m),
            start, length, num_m = low, win_len, principle

    if start == first[0] and length == win_len:  # sliding_result only contains one element.
        yield start, length, tuple(num_m)
