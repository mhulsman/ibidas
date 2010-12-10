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
            self.graph.addEdge(query_graph.Edge(slice,node,"param",pos))
        self.queue.extend(node._slices)

    def visitDataSlice(self,node):
        pass

    def visitUnaryOpSlice(self,node):
        self.graph.addNode(node.source)
        self.graph.addEdge(query_graph.Edge(node.source,node,"param",0))
        self.queue.append(node.source)
    
    def visitMultiOpSlice(self,node):
        for pos, slice in enumerate(node.sources):
            self.graph.addNode(slice)
            self.graph.addEdge(query_graph.Edge(slice,node,"param",pos))
        self.queue.extend(node.sources)
    
