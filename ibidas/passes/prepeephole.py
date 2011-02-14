import manager
import create_graph
import annotate_replinks
from ..utils import util
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, F_CACHE
from ..constants import *
_delay_import_(globals(),"..query_graph")
_delay_import_(globals(),"..ops")

class PrePeepHole(VisitorFactory(prefixes=("visit",), flags=NF_ERROR), manager.Pass):
    after=set([annotate_replinks.AnnotateRepLinks])

    @classmethod
    def run(cls, query, run_manager):
        self = cls()
        self.graph = run_manager.pass_results[create_graph.CreateGraph]

        for node in list(self.graph.nodes):
            self.visit(node)
    
    def visitNode(self, node):
        pass

    def visitChangeBookmarkOp(self, node):
        source = self.graph.getDataEdge(node).source
        target_edges = list(self.graph.edge_source[node])
        self.graph.dropNode(node)
        for target_edge in target_edges:
            assert isinstance(target_edge,query_graph.ParamEdge), "Unknown edge type encountered"
            target_edge.source = source
            self.graph.addEdge(target_edge)



