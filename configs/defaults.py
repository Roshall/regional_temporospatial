from yacs.config import CfgNode as CN


_C = CN()

_C.INDEX = CN()
_C.INDEX.SEARCH_METHOD = CN()
_C.INDEX.SEARCH_METHOD.TEMPO = 'FuzzyInnerAll'
_C.INDEX.SEARCH_METHOD.REGION = 'NotSureSearch'
_C.INDEX.SEARCH_METHOD.USER = 'NotSureSearch'

_C.INDEX.REGION = CN()
# should the region index simply return all region or
# distinguish between regions surely in the query region from others
_C.INDEX.REGION.COARSE = False
_C.INDEX.REGION.TYPE = 'grid'

_C.INDEX.REGION.GRID = CN()
_C.INDEX.REGION.GRID.SPACE = (8, 6)

_C.QUERY = 'index_one_pass'
# _C.QUERY = 'index_base'
# _C.QUERY = 'sliding_base'
