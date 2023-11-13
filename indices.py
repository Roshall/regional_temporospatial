import btree

from region import GridRegion
from functools import partial

from two_level_merge_index import TwoLevelMergeIndex, fuzzy

# key: (duration, begin)
# Here we need duration to deduct the correct beginning of a segment
# overlapping within the searching interval. Therefore, fuzzy search is needed.
Tempo2DIdx = partial(fuzzy(TwoLevelMergeIndex), btree.BtreeMap, btree.BtreeMultiMap)
# key: (duration, (x, y))
Out3DRegion = partial(TwoLevelMergeIndex, btree.BtreeMap, GridRegion)
# key:((duration, (x, y)), (duration, begin))
UserIdx = partial(TwoLevelMergeIndex, Out3DRegion, Tempo2DIdx)



