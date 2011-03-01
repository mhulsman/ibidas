import manager
import create_graph
import wrapper_planner
import serialize_exec

from ..utils import util
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, F_CACHE
from ..constants import *
_delay_import_(globals(),"..query_graph")
_delay_import_(globals(),"..ops")
_delay_import_(globals(),"..itypes","dimpaths")

class PythonPeepHole(VisitorFactory(prefixes=("visit",), flags=NF_ERROR), manager.Pass):
    after=set([wrapper_planner.WrapperPlanner])
    before=set([serialize_exec.SerializeExec])

    @classmethod
    def run(cls, query, run_manager):
        self = cls()
        self.graph = run_manager.pass_results[create_graph.CreateGraph]

        for node in list(self.graph.nodes):
            self.visit(node)
    
    def visitNode(self, node):
        pass

    def visitBroadcastOp(self, node):
        if not node.partial:
            return
        nplans = dimpaths.processPartial(node.bcdims, node.plan)
        updated=False
        bcpos = 0
        for oplan,nplan in zip(node.plan, nplans):
            if oplan != nplan:
                self.remove_compare_slice(node,bcpos)
                updated=True
            elif oplan ==  BCEXIST:
                bcpos += 1
        if updated:
            if bcpos == 0:
                self.remove_unaryop(node) 
            else:
                node = self.graph.copyNode(node)
                node.plan = nplans

    def remove_compare_slice(self, node, pos):
        for edge in list(self.graph.edge_target[node]):
            if edge.name == "compare_slices":
                if edge.pos == pos:
                    self.graph.dropEdge(edge)
                elif edge.pos >= pos:
                    edge.pos -= 1

            
    def remove_unaryop(self, node):
        source = self.graph.getDataEdge(node).source
        target_edges = list(self.graph.edge_source[node])
        self.graph.dropNode(node)
        for target_edge in target_edges:
            assert isinstance(target_edge,query_graph.ParamEdge), "Unknown edge type encountered"
            target_edge.source = source
            self.graph.addEdge(target_edge)


