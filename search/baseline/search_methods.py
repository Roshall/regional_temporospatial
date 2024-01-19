from functools import partial

from search.rest import sliding_window, expend
from search.verifier import obj_verify, len_filter
from search.baseline.naive import NaiveSliding
from utilities.data_preprocessing import group_by_frame


def sliding_framework(df, region, labels, duration, interval, *, method='base'):
    dur = duration[0]
    frames = group_by_frame(df, interval)
    obj_verifier = partial(obj_verify, labels)
    windows = sliding_window(frames, dur)
    if method == 'base':
        sliding = NaiveSliding(windows, dur, region.enclose, obj_verifier, len_filter, tuple(labels))
    else:
        sliding = None
    return expend(sliding, obj_verifier)
