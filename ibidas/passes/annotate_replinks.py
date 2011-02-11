import manager
import ensure_info
import create_graph
from ..utils import util
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, F_CACHE
from ..constants import *
_delay_import_(globals(),"..representor")
_delay_import_(globals(),"..ops")

class Expression(object):
    __slots__ = ["etype","eobj","in_slices","out_slices","between_slices"]
    def __init__(self, etype, eobj):
        self.etype = etype
        self.eobj = eobj
        self.in_slices = set()
        self.out_slices = set()
        self.between_slices = set()

    def addInSlice(self, in_slice):
        self.in_slices.add(in_slice)

    def addOutSlice(self, out_slice):
        self.out_slices.add(out_slice)

    def addBetweenSlice(self, between_slice):
        self.between_slices.add(between_slice)

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

        self.expressions = {}
        self.distribute(self.graph.root)
        self.graph.checkGraph()
        return self.expressions

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
            self.links[node] = lastlink
            selflink = lastlink
        
        if(not id(selflink) in self.expressions):
            self.expressions[id(selflink)] = Expression(selflink.__class__.__name__,selflink)
        e = self.expressions[id(selflink)]

        if(selflink is lastlink):
            e.addBetweenSlice(node)
        else:
            e.addOutSlice(node)
      
        for edge in self.graph.edge_target[node]:
            sourcelink = self.distribute(edge.source, selflink)
            if not sourcelink is selflink:
                e.addInSlice(edge.source)
                
        return selflink
        

     
