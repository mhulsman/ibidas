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


    def visitFilterOp(self, node):
        if node.has_missing:
            return

        constraint_edge = self.graph.getDataEdge(node, name="constraint")
        constraint = constraint_edge.source
        if len(constraint.dims) > 0:
            return

        pack_node = self.graph.getDataEdge(node).source
        if not isinstance(pack_node, ops.PackArrayOp):
            return 
        if len(self.graph.edge_source[pack_node]) > 1:
            return
        source = self.graph.getDataEdge(pack_node).source
        
        target_edges = list(self.graph.edge_source[node])
        if len(target_edges)!= 1:
            unpack_node = None
        else:
            target = target_edges[0].target
            if not isinstance(target, ops.UnpackArrayOp):
                unpack_node = None
            else:
                unpack_node = target

        if unpack_node is None:
            last_node = node
        else:
            last_node = unpack_node
      
        source_edge = self.graph.getDataEdge(pack_node)
        self.graph.dropEdge(source_edge)

        nslice = ops.CombinedUnaryUnaryOp(source, [pack_node, node, unpack_node], 'PackFilterUnpackOp')
        self.graph.addNode(nslice)
        self.graph.addEdge(query_graph.ParamEdge(source, nslice, "slice"))

        self.graph.dropEdge(constraint_edge)
        constraint_edge.target = nslice
        self.graph.addEdge(constraint_edge)

        ttarget_edges = list(self.graph.edge_source[last_node])
        for edge in ttarget_edges:
            self.graph.dropEdge(edge)
            edge.source = nslice
            self.graph.addEdge(edge)
        
        self.graph.dropNode(pack_node)
        self.graph.dropNode(node)
        if unpack_node:
            self.graph.dropNode(unpack_node)
       

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


