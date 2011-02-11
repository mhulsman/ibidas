import manager
import ensure_info
import create_graph
from ..utils import util
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, F_CACHE
from ..constants import *
_delay_import_(globals(),"..representor")
_delay_import_(globals(),"..ops")

class WrapperPlanner(VisitorFactory(prefixes=("link","distribute"), 
                                      flags=NF_ERROR | F_CACHE), manager.Pass):
    after=set([create_graph.CreateGraph])

    @classmethod
    def run(cls, query, run_manager, debug_mode):
        self = cls()
        self.graph = run_manager.pass_results[create_graph.CreateGraph]

        passes = set()
        for node in self.graph.nodes:
            if hasattr(node.__class__, "passes"):
                passes.update(node.passes)
        
        for p in passes:
            run_manager.add_pass(p, debug_mode=debug_mode)

