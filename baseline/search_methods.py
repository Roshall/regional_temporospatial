from functools import partial

from baseline.helper import obj_verify, len_filter, expend
from baseline.naive import NaiveSliding


def naive_sliding(window, region, labels, duration):
    obj_verifier = partial(obj_verify, labels)
    sliding = NaiveSliding(window, duration, region.enclose, obj_verifier, len_filter, tuple(labels))
    return expend(sliding, obj_verifier)