from functools import partial

from search.baseline.co_moving import CoMovementPattern
from search.baseline.helper import obj_verify, len_filter, expend, sliding_window
from search.baseline.naive import NaiveSliding
from utilities.box2D import Box2D
from utilities.data_preprocessing import group_by_frame
from utilities.dataset import load_fake


class TestSliding:
    data, _, _ = load_fake()
    interval = 0, 10
    frames = group_by_frame(data, interval)
    region = Box2D((0, 4, 0, 5))
    labels = {0: 1}
    duration = 3
    obj_verifier = partial(obj_verify, labels)
    windows = sliding_window(frames, duration)

    def test_naive(self):
        sliding = NaiveSliding(self.windows, self.duration, self.region.enclose,
                               self.obj_verifier, len_filter, tuple(self.labels))
        res = list(sliding)
        assert len(res) == 8

    def test_expend(self):
        ans = [CoMovementPattern(labels={1: 0, 3: 0, 2: 0}, interval=[5, 10]),
               CoMovementPattern(labels={1: 0, 3: 0}, interval=[4, 10]),
               CoMovementPattern(labels={1: 0}, interval=[1, 10])]
        sliding = NaiveSliding(self.windows, self.duration, self.region.enclose,
                               self.obj_verifier, len_filter, tuple(self.labels))
        res = list(expend(sliding, self.obj_verifier))

        assert res == ans

