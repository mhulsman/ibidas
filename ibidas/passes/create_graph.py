import manager
import ensure_info
from ..utils.multi_visitor import VisitorFactory, NF_ELSE
from .. import query_graph

_delay_import_(globals(),"..repops")

class CreateGraph(VisitorFactory(prefixes=("visit",), 
                                      flags=NF_ELSE), manager.Pass):
    after=set([ensure_info.EnsureInfo])
    @classmethod
    def run(cls, query, run_manager):
        self = cls()

        visited = set()
        self.queue.append(query.root)
        self.graph.setRoot(query.root)
        self.graph.addNode(query.root)
        
        while(self.queue):
            elem = self.queue.pop()
            if(elem in visited):
                continue
            self.visit(elem)
            visited.add(elem)

        return self.graph

    def __init__(self):
        self.queue = []
        self.graph = query_graph.Graph()
   
    def visitelse(self,node):
        raise RuntimeError, "Unknown type of node encountered"
    
    def visitFixate(self,node):
        for pos, slice in enumerate(node._slices):
            self.graph.addNode(slice)
            self.graph.addEdge(query_graph.Edge(slice,node,"paramlist","slices",pos))
        self.queue.extend(node._slices)

    def visitDataSlice(self,node):
        pass

    def visitUnaryOpSlice(self,node):
        self.graph.addNode(node.source)
        self.graph.addEdge(query_graph.Edge(node.source,node,"param","slice",0))
        self.queue.append(node.source)
    
    def visitMultiOpSlice(self,node):
        for pos, slice in enumerate(node.sources):
            self.graph.addNode(slice)
            self.graph.addEdge(query_graph.Edge(slice,node,"paramlist","slices",pos))
        self.queue.extend(node.sources)
    
    def visitFilterSlice(self,node):
        self.visitUnaryOpSlice(node)
        self.graph.addNode(node.constraint)
        self.graph.addEdge(query_graph.Edge(node.constraint,node,"param","constraint",0))
        self.queue.append(node.constraint)

   
    def visitEnsureCommonDimSlice(self,node):
        self.visitUnaryOpSlice(node)
        for slice in node.refslices:
            self.graph.addNode(slice)
            self.graph.addEdge(query_graph.Edge(slice,node,"paramchoice","compare_slice",0))
        self.queue.extend(node.refslices)
     
    def visitBroadcastSlice(self,node):
        self.visitUnaryOpSlice(node)
        for pos, slicelist in enumerate(node.refsliceslist):
            for slice in slicelist:
                self.graph.addNode(slice)
                self.graph.addEdge(query_graph.Edge(slice,node,"paramchoicelist","compare_slices",pos))
            self.queue.extend(slicelist)
       
