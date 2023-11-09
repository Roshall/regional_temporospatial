from db_interface import Index


class TwoLevelMergeIndex(Index):
    def __init__(self, outer_cls, inner_cls):
        self._outer = outer_cls()
        self._inner_cls = inner_cls

    def add(self, key, entry) -> None:
        if (inner_idx := self._outer.where_contain(key[0])) is None:
            inner_idx = self._inner_cls()
            self._outer.add(key[0], inner_idx)
        inner_idx.add(key[1], entry)

    def where_contain(self, key):
        return (inner := self._outer.where_contain(key[0])) and inner.where_contain(key[1])

    def where_intersect(self, bboxes):
        for inner_idx in self._outer.where_intersect(bboxes[0]):
            yield from inner_idx.where_intersect(bboxes[1])
