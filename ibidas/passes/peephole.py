import manager
from ..utils.multi_visitor import VisitorFactory, F_CACHE, NF_ROBJ, NV_ELSE


class PeepHoleOptimizer(VisitorFactory(prefixes=("optimize", "func"), 
                                      flags=NF_ELSE), 
                       pass_manager.Pass):

    @classmethod
    def run(cls, query, pass_results):
        self = cls()

