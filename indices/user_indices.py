from functools import partial

import btree

from .region import GridRegion, unsure
from .two_level_merge_index import TwoLevelMergeIndex

# Here we need duration to deduct the correct beginning of a segment
# overlapping within the searching interval. Therefore, fuzzy search is needed.
# key: (duration, begin)
Tempo2DIdx = partial(TwoLevelMergeIndex, btree.BtreeMap, btree.BtreeMultiMap, 'FuzzyInnerAll')
# key: ((x, y), duration)
Out3DRegion = partial(TwoLevelMergeIndex, unsure(GridRegion), btree.BtreeMap, 'NotSureSearch')
# key:(((x, y), duration), (duration, begin))
UserIdx = partial(TwoLevelMergeIndex, Out3DRegion, Tempo2DIdx, 'NotSureSearch')



