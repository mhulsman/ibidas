import manager
from ..utils.multi_visitor import VisitorFactory, NF_ERROR
from .. import query_graph
from ..constants import *
_delay_import_(globals(),"..repops")
_delay_import_(globals(),"..repops_slice")
_delay_import_(globals(),"..slices")

class EnsureInfo(VisitorFactory(prefixes=("findFirstKnown","processQuery"), 
                                      flags=NF_ERROR), manager.Pass):
    @classmethod
    def run(cls, query, run_manager):
        query_root = query.root
        while(not query_root._state & RS_SLICES_KNOWN):
            self = cls()
            first_known_nodes = self.findFirstKnown(query.root)
            temp_root = repops_slice.Combine(*first_known_nodes)
            temp_root = repops.ApplyFuncRep(temp_root,apply_slice,slices.DetectTypeSlice,None)
            inferred_slices = root._getResultSlices()

            self.inferred_node_slices = dict()
            for node in first_known_nodes:
                nslices = len(node._slices)
                self.inferred_node_slices = known_slices[:nslices]
                known_slices = known_slices[nslices:]
        
            query_root = self.processQuery(query_root)
        return query_root

    def findFirstKnownRepresentor(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return (self,)
        else:
            raise RuntimeError, "Cannot find known slices for " + str(type(node))

    def findFirstKnownUnaryOpRep(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return (self,)
        else:
            return self.findFirstKnown(node._source)
    
    def findFirstKnownMultiOpRep(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return (self,)
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
            nnode.initalize(self.inferred_node_slices[node],RS_ALL_KNOWN | RS_INFERRED)
            return nnode
        else:
            raise RuntimeError, "Cannot find known slices for " + str(type(node))

    def processQueryUnaryOpRep(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return self.processQueryRepresentor(nnode)
        else:
            nnode = self.getNodeCopy(node)
            nnode._source = self.processQuery(node._source)
            nnode.process()
            return nnode
    
    def processQueryMultiOpRep(self,node):
        if(node._state & RS_SLICES_KNOWN):
            return self.processQueryRepresentor(nnode)
        else:
            nnode = self.getNodeCopy(node)
            nnode._sources = tuple([self.processQuery(source) for source in node._sources])
            nnode.process()
            return nnode

