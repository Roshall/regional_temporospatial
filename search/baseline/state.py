from collections.abc import Iterable
from itertools import islice

from search.co_moving import CoMovementPattern
from search.rest import state_sliding


def state_slider(frames: Iterable, win_len, obj_verifier, dfilter):
    state_maintainer = state_maintain(win_len)
    pat_iter = pat_wrapper(frames, obj_verifier, dfilter)
    return state_sliding(pat_iter, obj_verifier, state_maintainer)


def pat_wrapper(frames, obj_verifier, dfilter):
    for fid, objs in frames:
        objs = dfilter(objs)
        objs = objs.set_index('oid')['cls'].to_dict()['cls']
        if obj_verifier(objs.keys()):
            yield CoMovementPattern(objs, [fid, fid])


def state_maintain(win_len):
    valid_ptr: int | None = None

    def inner(cur, prev, new, count, absort):
        nonlocal valid_ptr
        prev_iter = iter(prev)
        fruits = []
        if valid_ptr is not None:
            if count >= valid_ptr:
                if count != valid_ptr:
                    fruits = islice(prev_iter, valid_ptr, count)
                valid_ptr = len(new) if count != len(prev) else None
            elif absort:
                valid_ptr -= count + len(new)
                if cur.end - prev[valid_ptr - 1].start + 1 == win_len:
                    valid_ptr -= 1
            else:
                fruits = islice(prev_iter, valid_ptr, None)
                valid_ptr = None
        else:
            if cur.end - prev[-1].start + 1 == win_len:
                valid_ptr = len(new) + len(prev) - count - 1
        return fruits, prev_iter

    return inner
