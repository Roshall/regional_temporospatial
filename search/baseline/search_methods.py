from functools import partial

from search.rest import sliding_window, expend
from search.verifier import obj_verify, len_filter
from search.baseline.naive import NaiveSliding
from utilities.data_preprocessing import group_by_frame


def sliding_framework(df, region, labels, duration, interval, *, method='base'):
    frames = group_by_frame(df, interval)
    obj_verifier = partial(obj_verify, labels)
    windows = sliding_window(frames, duration)
    if method == 'base':
        sliding = NaiveSliding(windows, duration, region.enclose, obj_verifier, len_filter, tuple(labels))
    else:
        sliding = None
    return expend(sliding, obj_verifier)
