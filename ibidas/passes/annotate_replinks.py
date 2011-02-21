import manager
import ensure_info
import create_graph
from ..utils import util
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, F_CACHE
from ..constants import *
_delay_import_(globals(),"..representor")



class AnnotateRepLinks(VisitorFactory(prefixes=("link","distribute"), 
                                      flags=NF_ERROR | F_CACHE), manager.Pass):
    after=set([ensure_info.EnsureInfo,create_graph.CreateGraph])

    @classmethod
    def run(cls, query, run_manager):
        self = cls()
        query_root = query.root
        self.graph = run_manager.pass_results[create_graph.CreateGraph]
        self.links = self.graph.na['links']
        self.link(query_root)
        self.distribute(self.graph.root)
        return self.links

    def linkRepresentor(self,node):
        for slice in node._slices:
            if not slice in self.links and slice in self.graph.nodes:
                self.links[slice] = node

    def linkUnaryOpRep(self,node):
        self.link(node._source)
        self.linkRepresentor(node)
    
    def linkProject(self,node):
        self.link(node._source)
        if(node._project_sources):
            for name, elem in node._project_sources:
                if(isinstance(elem, representor.Representor)):
                    self.link(elem)
   
    def linkMultiOpRep(self,node):
        for source in node._sources:
            self.link(source)
        self.linkRepresentor(node)


    def distributeOp(self, node, lastlink=None):
        if(node in self.links):
            selflink = self.links[node]
        else:
            selflink = lastlink

        for edge in self.graph.edge_target[node]:
            sourcelink = self.distribute(edge.source, selflink)
                
        self.links[node] = selflink
        return selflink

