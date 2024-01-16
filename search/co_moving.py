from collections import Counter
from collections.abc import Mapping, MutableSequence
from dataclasses import dataclass


@dataclass(slots=True)
class CoMovementPattern:
    labels: Mapping  # dict of id -> label
    interval: MutableSequence = None

    @property
    def objs(self):
        return self.labels.keys()

    def label_count(self):
        return Counter(self.labels.values())

    def update_end(self, end: int):
        self.interval[1] = end
        return self

    def __len__(self):
        return len(self.labels)

    def __and__(self, other):
        intersect_objs = self.objs & other.objs
        cls = type(self)
        return cls({obj: self.labels[obj] for obj in intersect_objs})

    def __eq__(self, other):
        return tuple(self.interval) == tuple(other.interval) and self.objs == other.objs

    def __str__(self):
        return f'({tuple(self.labels)}, {self.interval})'
