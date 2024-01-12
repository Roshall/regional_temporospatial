from collections import Counter
from functools import partial

from search.baseline.co_moving import CoMovementPattern


class NaiveSliding:
    def __init__(self, windows, win_len, region_verifier, obj_verifier, len_filter, target_label):
        self.wins_iter = windows
        self.olen_m, self.label_m = Counter(), {}
        self.reg_ver = region_verifier
        self.len_filter = partial(len_filter, self.olen_m, win_len)
        self.obj_ver = obj_verifier
        self.target_label = target_label
        self.win_len = win_len

    def _update(self, objs):
        objs = objs[objs['cls'].isin(self.target_label)]  # class verification
        objs = objs[self.reg_ver(objs[['x', 'y']])]  # region verification
        self.olen_m.update(objs['oid'])
        self.label_m.update(objs.set_index('oid')['cls'])

    def _filter(self, low, high):
        ids = self.len_filter()
        if ids:
            candi = CoMovementPattern({id_: self.label_m[id_] for id_ in ids})
            if self.obj_ver(candi.label_count()) and high - low == self.win_len - 1:
                candi.interval = [low, high]
                return candi
            else:
                return False
        else:
            return False

    def _subtract(self, abandoned):
        self.olen_m.subtract(abandoned)
        desolated = [obj for obj, num in self.olen_m.items() if num <= 0]
        for obj in desolated:
            del self.olen_m[obj]
            del self.label_m[obj]

    def __iter__(self):
        if (win := next(self.wins_iter, None)) is None:
            return
        low, high = win[0][0], win[-1][0]
        for _, objs in win:
            self._update(objs)
        if candi := self._filter(low, high):
            yield candi

        abandoned = win[0][1]['oid']
        for win in self.wins_iter:
            self._subtract(abandoned[abandoned.isin(self.olen_m)])

            fid, objs = win[-1]
            low = win[0][0]
            self._update(objs)
            if candi := self._filter(low, fid):
                yield candi
            abandoned = win[0][1]['oid']
