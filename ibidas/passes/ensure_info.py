import copy
import manager
from ..utils import util
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, F_CACHE
from .. import query_graph
from ..constants import *
_delay_import_(globals(),"..repops")
_delay_import_(globals(),"..repops_multi")
_delay_import_(globals(),"..representor")
_delay_import_(globals(),"..ops")

class EnsureInfo(VisitorFactory(prefixes=("findFirstKnown","processQuery"), 
                                      flags=NF_ERROR | F_CACHE), manager.Pass):
    @classmethod
    def run(cls, query, run_manager):
        query_root = query.root
        if(not query_root._slicesKnown()):
            while(not query_root._slicesKnown()):
                self = cls()
                self.copied_nodes = dict()
                first_known_nodes = self.findFirstKnown(query_root)
                temp_root = repops_multi.Combine(*first_known_nodes)
                temp_root = repops.ApplyFuncRep(temp_root,repops.apply_slice,ops.DetectTypeOp,None)
                known_slices = temp_root._getResultSlices(endpoint=False)

                self.inferred_node_slices = dict()
                for node in first_known_nodes:
                    nslices = len(node._slices)
                    self.inferred_node_slices[id(node)] = known_slices[:nslices]
                    known_slices = known_slices[nslices:]
                query_root = self.processQuery(query_root)
            query.root = query_root
        return query.root

    def findFirstKnownRepresentor(self,node):
        if node._slicesKnown():
            return (node,)
        else:
            raise RuntimeError, "Cannot find known slices for " + str(type(node))

    def findFirstKnownUnaryOpRep(self,node):
        if node._slicesKnown():
            return (node,)
        else:
            return self.findFirstKnown(node._source)
    
    def findFirstKnownProject(self,node):
        if node._slicesKnown():
            return (node,)
        else:
            x = list(self.findFirstKnown(node._source))
            if(node._project_sources):
               for name, elem in node._project_sources:
                    if(isinstance(elem, representor.Representor)):
                        x.extend(self.findFirstKnown(elem))
            return tuple(x)
   
    def findFirstKnownMultiOpRep(self,node):
        if node._slicesKnown():
            return (node,)
        else:
            return sum([self.findFirstKnown(source) for source in node._sources],())

    def getNodeCopy(self,node):
        if(not id(node) in self.copied_nodes):
            self.copied_nodes[id(node)] = copy.copy(node)
        return self.copied_nodes[id(node)]

    def processQueryRepresentor(self,node):
        if node._slicesKnown():
            assert id(node) in self.inferred_node_slices, "Cannot find node in inferred nodes map"
            nnode = self.getNodeCopy(node)
            nnode._initialize(self.inferred_node_slices[id(node)])
            return nnode
        else:
            raise RuntimeError, "Cannot find known slices for " + str(type(node))

    def processQueryUnaryOpRep(self,node):
        if node._slicesKnown():
            return self.processQueryRepresentor(node)
        else:
            nnode = self.getNodeCopy(node)
            nnode._source = self.processQuery(node._source)
            nnode._process(nnode._source,*node._params[0], **node._params[1])
            return nnode
    
    def processQueryProject(self,node):
        if node._slicesKnown():
            return self.processQueryRepresentor(node)
        else:
            nnode = self.getNodeCopy(node)
            nnode._source = self.processQuery(node._source)
            if(nnode._project_sources):
                npsources = []
                for name,elem in nnode._project_sources:
                     if(isinstance(elem,representor.Representor)):
                        elem = self.processQuery(elem)
                     npsources.append((name,elem))
                nnode._project_sources = npsources
            nnode._process(nnode._source,*node._params[0], **node._params[1])
            return nnode
   
    def processQueryMultiOpRep(self,node):
        if node._slicesKnown():
            return self.processQueryRepresentor(node)
        else:
            nnode = self.getNodeCopy(node)
            nnode._sources = tuple([self.processQuery(source) for source in node._sources])
            nnode._process(nnode._sources,*node._params[0], **node._params[1])
            return nnode

    def processQueryMatch(self, node):
        if node._slicesKnown():
            return self.processQueryRepresentor(node)
        else:
            nnode = self.getNodeCopy(node)
            lsource, rsource, lslice, rslice = node._sources
            if(lslice is None or rslice is None):
                lsource = self.processQuery(lsource)
                rsource = self.processQuery(rsource)
                nnode._process((lsource,rsource,lslice,rslice),*node._params[0], **node._params[1])
                lsource, rsource, lslice, rslice = node._sources

            assert not lslice is None and not rslice is None, "Cannot determine left and/or right slice for Match operation"
            nnode._sources = tuple([self.processQuery(source) for source in node._sources])
            nnode._process(nnode._sources,*node._params[0], **node._params[1])
            return nnode

