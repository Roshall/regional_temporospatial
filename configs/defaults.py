from yacs.config import CfgNode as CN


_C = CN()

_C.INDEX = CN()
_C.INDEX.SEARCH_METHOD = CN()
_C.INDEX.SEARCH_METHOD.TEMPO = 'FuzzyInnerAll'
_C.INDEX.SEARCH_METHOD.REGION = 'NotSureSearch'
_C.INDEX.SEARCH_METHOD.USER = 'NotSureSearch'

_C.INDEX.REGION = CN()
_C.INDEX.REGION.SURE = False
