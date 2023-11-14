from functools import partial

import btree

from .region import GridRegion
from .two_level_merge_index import TwoLevelMergeIndex, fuzzy

# Here we need duration to deduct the correct beginning of a segment
# overlapping within the searching interval. Therefore, fuzzy search is needed.
# key: (duration, begin)
Tempo2DIdx = partial(fuzzy(TwoLevelMergeIndex), btree.BtreeMap, btree.BtreeMultiMap)
# key: (duration, (x, y))
Out3DRegion = partial(TwoLevelMergeIndex, btree.BtreeMap, GridRegion)
# key:((duration, (x, y)), (duration, begin))
UserIdx = partial(TwoLevelMergeIndex, Out3DRegion, Tempo2DIdx)



