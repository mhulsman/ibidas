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

    def remove_unaryop(self, node):
        source = self.graph.getDataEdge(node).source
        target_edges = list(self.graph.edge_source[node])
        self.graph.dropNode(node)
        for target_edge in target_edges:
            assert isinstance(target_edge,query_graph.ParamEdge), "Unknown edge type encountered"
            target_edge.source = source
            self.graph.addEdge(target_edge)
    visitChangeBookmarkOp=remove_unaryop
    visitChangeNameOp=remove_unaryop
    visitChangeDimPathOp=remove_unaryop

    def visitPackArrayOp(self,node):
        target_edges = list(self.graph.edge_source[node])
        if(len(target_edges) != 1): 
            return

        target = target_edges[0].target
        if(not isinstance(target, ops.UnpackArrayOp)):
            return
        
        pack_depth = len(node.pack_dims)
        unpack_depth = len(target.unpack_dims)
        return self.combine_pack_unpack(pack_depth, unpack_depth, node, target)

    def visitUnpackArrayOp(self,node):
        target_edges = list(self.graph.edge_source[node])
        if(len(target_edges) != 1): 
            return

        target = target_edges[0].target
        if(not isinstance(target, ops.PackArrayOp)):
            return
        
        unpack_depth = len(node.unpack_dims)
        pack_depth = len(target.pack_dims)
        return self.combine_pack_unpack(pack_depth, unpack_depth, node, target)

    def combine_pack_unpack(self, pack_depth, unpack_depth, node, target):
        sourceedge = self.graph.getDataEdge(node)
        self.graph.dropEdge(sourceedge)
        if(pack_depth != unpack_depth): 
            if(pack_depth > unpack_depth):
                nsource = ops.PackArrayOp(sourceedge.source,pack_depth - unpack_depth)
            else:
                nsource = ops.UnpackArrayOp(sourceedge.source,unpack_depth - pack_depth)
            self.graph.addNode(nsource)
            self.graph.addEdge(query_graph.ParamEdge(sourceedge.source,nsource,"slice"))
        else:
            nsource = sourceedge.source
        
        ttarget_edges = list(self.graph.edge_source[target])
        for edge in ttarget_edges:
            self.graph.dropEdge(edge)
            edge.source = nsource
            self.graph.addEdge(edge)

        self.graph.dropNode(node)
        self.graph.dropNode(target)
           

        
