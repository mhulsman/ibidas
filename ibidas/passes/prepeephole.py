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

        modified = True
        while modified:
            modified = False
            for node in list(self.graph.nodes):
                modified = modified or self.visit(node)
    
    def visitNode(self, node):
        return False

    def remove_unaryop(self, node):
        source = self.graph.getDataEdge(node).source
        target_edges = list(self.graph.edge_source[node])
        self.graph.dropNode(node)
        for target_edge in target_edges:
            assert isinstance(target_edge,query_graph.ParamEdge), "Unknown edge type encountered"
            target_edge.source = source
            self.graph.addEdge(target_edge)
        return True
    visitNoOp=remove_unaryop
    visitChangeBookmarkOp=remove_unaryop
    visitChangeNameOp=remove_unaryop
    visitChangeDimPathOp=remove_unaryop

    def visitPackArrayOp(self,node):
        target_edges = list(self.graph.edge_source[node])

        unpack_depth = []
        targets = []
        for target_edge in target_edges:
            target = target_edge.target
            if(not isinstance(target, ops.UnpackArrayOp)):
                return False
            unpack_depth.append(len(target.unpack_dims))
            targets.append(target)
       
        #could be relaxed if support is included in combine_pack_unpack
        if not len(set(unpack_depth)) == 1:
            return False

        pack_depth = len(node.pack_dims)
        unpack_depth = unpack_depth.pop()
        return self.combine_pack_unpack(pack_depth, unpack_depth, node, targets)

    def visitUnpackArrayOp(self,node):
        target_edges = list(self.graph.edge_source[node])

        pack_depth = []
        targets = []
        for target_edge in target_edges:
            target = target_edge.target
            if(not isinstance(target, ops.PackArrayOp)):
                return False
            pack_depth.append(len(target.pack_dims))
            targets.append(target)
        
        #could be relaxed if support is included in combine_pack_unpack
        if not len(set(pack_depth))  == 1:
            return False

        unpack_depth = len(node.unpack_dims)
        pack_depth = pack_depth.pop()
        return self.combine_pack_unpack(pack_depth, unpack_depth, node, targets)

    def combine_pack_unpack(self, pack_depth, unpack_depth, node, targets):
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
       
        for target in targets:
            ttarget_edges = list(self.graph.edge_source[target])
            for edge in ttarget_edges:
                self.graph.dropEdge(edge)
                edge.source = nsource
                self.graph.addEdge(edge)
            
            self.graph.dropNode(target)

        self.graph.dropNode(node)
        return True           

        
