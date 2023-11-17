import sys
from abc import ABCMeta, abstractmethod


class TwoLevelMergeIndex:
    def __init__(self, outer_cls, inner_cls, intersect_strategy='ReturnEntry'):
        self._outer = outer_cls()
        self._inner_cls = inner_cls
        self.where_intersect = strategies[intersect_strategy](self._outer)

    def add(self, key, entry) -> None:
        if (inner_idx := self._outer.where_contain(key[0])) is None:
            inner_idx = self._inner_cls()
            self._outer.add(key[0], inner_idx)
        inner_idx.add(key[1], entry)

    def where_contain(self, key):
        return (inner := self._outer.where_contain(key[0])) and inner.where_contain(key[1])


class IntersectStrategy(metaclass=ABCMeta):
    def __init__(self, outer_idx):
        self._outer = outer_idx

    @abstractmethod
    def __call__(self, bboxes):
        """
        return items according to a concrete strategy
        :param bboxes: (dim0_min, dim0_max, dim1_min, dim1_max, ...)
        :return: iterable itemes
        """
        pass


class IntersectReturnEntry(IntersectStrategy):
    """
    get inner entries one by one
    """
    def __init__(self, outer_idx):
        super().__init__(outer_idx)

    def __call__(self, bboxes):
        for inner_idx in self._outer.where_intersect(bboxes[0]):
            yield from inner_idx.where_intersect(bboxes[1])


class IntersectFuzzySearch(IntersectStrategy):
    """
    linear constrains search
    """
    def __init__(self, outer_idx):
        super().__init__(outer_idx)

    def __call__(self, bboxes):
        """
        :param bboxes: just care (dim2_min, dim2_max)
        :return: all possible region, iterable.
        """
        lo, hi = bboxes
        for outer_key, inner_idx in self._outer:
            yield from inner_idx.where_intersect((lo - outer_key + 1, hi))


class IntersectNotSureSearch(IntersectStrategy):
    """
    separate absolute inner entries from the unsure
    return iterable tuple of generators: (not_sure, sure)
    """
    def __init__(self, outer_idx):
        super().__init__(outer_idx)

    def __call__(self, bboxes):
        candidates, probation = self._outer.where_intersect(bboxes[0])
        return (entry for inner in candidates for entry in inner.where_intersect([bboxes[1]])), \
            (entry for inner in probation for entry in inner.where_intersect(bboxes[1]))


class IntersectFuzzyInnerAll(IntersectStrategy):
    """
    return iterable inner search result as a whole set with the linear constraints.
    """
    def __init__(self, outer_idx):
        super().__init__(outer_idx)

    def __call__(self, bboxes):
        lo, hi = bboxes
        for outer_key, inner_idx in self._outer:
            yield inner_idx.where_intersect((lo - outer_key + 1, hi))


implemented_strategies = ['ReturnEntry', 'FuzzySearch', 'NotSureSearch', 'FuzzyInnerAll']
strategies = {stra: getattr(sys.modules[__name__], f'Intersect{stra}') for stra in implemented_strategies}
