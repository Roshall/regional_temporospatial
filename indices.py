import btree

from region import GridRegion
from functools import partial

from two_level_merge_index import TwoLevelMergeIndex

# key: (duration, begin)
Tempo2DIdx = partial(TwoLevelMergeIndex, btree.BtreeMap, btree.BtreeMultiMap)
# key: (duration, (x, y))
Out3DRegion = partial(TwoLevelMergeIndex, btree.BtreeMap, GridRegion)
# key:((duration, (x, y)), (duration, begin))
user_idx = TwoLevelMergeIndex(Out3DRegion, Tempo2DIdx)



