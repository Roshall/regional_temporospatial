import btree

from db_interface import Index


class Tempo2DIndex(Index):
    def __init__(self):
        self._store = btree.BtreeMap()

    def add(self, key, entry):
        if (begin_idx := self._store.where_contain(key.duration)) is None:
            begin_idx = btree.BtreeMultiMap()
            self._store.add(begin_idx)
        begin_idx.add(key.begin, entry)

    def where_contain(self, key):
        return (inner := self._store.where_contain(key.duration)) and inner.where_contain(key.begin)

    def where_intersect(self, bbox):
        for beg_idx in self._store.where_intersect(bbox[:2]):
            yield from beg_idx.where_intersect(bbox[2:])
