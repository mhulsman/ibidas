from ibidas.utils import util
from ibidas.utils.config import config
from ibidas.pre import Pre
_delay_import_(globals(),"ibidas","*")

predefined_algs = Pre()


def Whiten(source, dim=None):
    return ((source - source.Mean(dim=dim)) / source.Std(dim=dim))
predefined_algs.register(Whiten,category="scaling")
