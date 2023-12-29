from collections import Counter
from functools import partial


class NaiveSliding:
    def __init__(self, windows, region_verifier, obj_verifier, len_verifier, target_label):
        self.wins_iter = windows
        self.olen_m, self.onum_m = Counter(), Counter()
        self.reg_ver = region_verifier
        self.len_ver = partial(len_verifier, self.onum_m, self.wins_iter.maxlen)
        self.obj_ver = partial(obj_verifier, self.onum_m)
        self.target_label = target_label
        self.win_len = 0

    def _update(self, objs):
        objs = objs[objs['cls'].isin(self.target_label)]  # class verification
        objs = objs[self.reg_ver(objs[['x', 'y']])]  # region verification
        self.onum_m.update(objs['cls'])
        self.olen_m.update(objs['oid'])

    def _has_new_result(self, low, high):
        if self.obj_ver() and high - low == self.win_len - 1 and self.len_ver:
            return True
        else:
            return False

    def __next__(self):
        if (win := next(self.wins_iter, None)) is None:
            return
        self.win_len = win.maxlen
        low, high = win[0][0], win[-1][0]
        for _, objs in self.wins_iter:
            self._update(objs)
        if self._has_new_result(low, high):
            yield low, self.olen_m.copy()

        abandoned = win[0][1]
        for win in self.wins_iter:
            self.onum_m.subtract(abandoned['cls'])
            self.olen_m.subtract(abandoned['oid'])

            fid, objs = win[-1]
            low = win[0][0]
            self._update(objs)
            if self._has_new_result(low, fid):
                yield low, self.olen_m.copy()
            abandoned = win[0][1]
