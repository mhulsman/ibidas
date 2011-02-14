import manager
import ensure_info
import create_graph
from ..utils import util
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, F_CACHE
from ..constants import *
from annot_objs import *
_delay_import_(globals(),"..representor")
_delay_import_(globals(),"..ops")



class AnnotateRepLinks(VisitorFactory(prefixes=("link","distribute","createExp"), 
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
            selflink = lastlink
        
        if(not id(selflink) in self.expressions):
            e = self.createExp(selflink)
            self.expressions[id(selflink)] = e
        else:
            e = self.expressions[id(selflink)]

        e.addAllSlice(node)
        if(not selflink is lastlink):
            e.addOutSlice(node)
      
        for edge in self.graph.edge_target[node]:
            sourcelink = self.distribute(edge.source, selflink)
            if not sourcelink is selflink:
                e.addInSlice(edge.target)
                
        self.links[node] = e
        return selflink
        

    def createExpRepresentor(self, rep):
       return Expression(rep.__class__.__name__, rep)

    def createExpFilter(self, rep):
       return FilterExpression(rep.__class__.__name__, rep)

    def createExpBinaryFuncElemOp(self, rep):
        return BinFuncElemExpression(rep.__class__.__name__, rep)
    
    def createExpMatch(self, rep):
        return MatchExpression(rep.__class__.__name__, rep)
