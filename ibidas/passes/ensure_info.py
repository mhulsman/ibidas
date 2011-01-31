import copy
import manager
from ..utils import util
from ..utils.multi_visitor import VisitorFactory, NF_ERROR
from .. import query_graph
from ..constants import *
_delay_import_(globals(),"..repops")
_delay_import_(globals(),"..repops_multi")
_delay_import_(globals(),"..slices")

class EnsureInfo(VisitorFactory(prefixes=("findFirstKnown","processQuery"), 
                                      flags=NF_ERROR), manager.Pass):
    @classmethod
    def run(cls, query, run_manager):
        query_root = query.root
        while(not query_root._state & RS_SLICES_KNOWN):
            self = cls()
            self.copied_nodes = dict()
            first_known_nodes = self.findFirstKnown(query.root)
            temp_root = repops_multi.Combine(first_known_nodes)
            temp_root = repops.ApplyFuncRep(temp_root,repops.apply_slice,slices.DetectTypeSlice,None)
            known_slices = temp_root._getResultSlices()

            self.inferred_node_slices = dict()
            for node in first_known_nodes:
                nslices = len(node._slices)
                self.inferred_node_slices[node] = known_slices[:nslices]
                known_slices = known_slices[nslices:]
        
            query_root = self.processQuery(query_root)
        query.root = query_root
        return query.root

    def findFirstKnownRepresentor(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return (node,)
        else:
            raise RuntimeError, "Cannot find known slices for " + str(type(node))

    def findFirstKnownUnaryOpRep(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return (node,)
        else:
            return self.findFirstKnown(node._source)
    
    def findFirstKnownMultiOpRep(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return (node,)
        else:
            return sum([self.findFirstKnown(source) for source in node._sources],())

    def getNodeCopy(self,node):
        if(not node in self.copied_nodes):
            self.copied_nodes[node] = copy.copy(node)
        return self.copied_nodes[node]

    def processQueryRepresentor(self,node):
        if(node._state & RS_SLICES_KNOWN):
            assert node in self.inferred_node_slices, "Cannot find node in inferred nodes map"
            nnode = self.getNodeCopy(node)
            nnode._initialize(self.inferred_node_slices[node],RS_ALL_KNOWN | RS_INFERRED)
            return nnode
        else:
            raise RuntimeError, "Cannot find known slices for " + str(type(node))

    def processQueryUnaryOpRep(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return self.processQueryRepresentor(node)
        else:
            nnode = self.getNodeCopy(node)
            nnode._source = self.processQuery(node._source)
            nnode._process(nnode._source,*node._params[0], **node._params[1])
            return nnode
    
    def processQueryMultiOpRep(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return self.processQueryRepresentor(node)
        else:
            nnode = self.getNodeCopy(node)
            nnode._sources = tuple([self.processQuery(source) for source in node._sources])
            nnode._process(nnode._sources,*node._params[0], **node._params[1])
            return nnode

