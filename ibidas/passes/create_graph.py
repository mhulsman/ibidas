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
        assert len(node._slices) == 1, "Query root should have one slice"
        self.graph.setRoot(node._slices[0])
        self.graph.addNode(node._slices[0])
        self.queue.append(node._slices[0])
        
    def visitExtendOp(self,node):
        if(hasattr(node,'create_graph_exec')):
            node.graph_exec(self)

    def visitDataOp(self,node):
        pass

    def visitUnaryUnaryOp(self,node):
        self.graph.addNode(node.source)
        self.graph.addEdge(query_graph.ParamEdge(node.source,node,"slice"))
        self.queue.append(node.source)
    
    def visitMultiUnaryOp(self,node):
        for pos, slice in enumerate(node.sources):
            self.graph.addNode(slice)
            self.graph.addEdge(query_graph.ParamListEdge(slice,node,"slices",pos))
        self.queue.extend(node.sources)

    def visitMultiMultiOp(self, node):
        for pos, slice in enumerate(node.sources):
            self.graph.addNode(slice)
            self.graph.addEdge(query_graph.ParamListEdge(slice,node,"slices",pos))
        self.queue.extend(node.sources)
        
    def visitFilterOp(self,node):
        self.visitUnaryUnaryOp(node)
        self.graph.addNode(node.constraint)
        self.graph.addEdge(query_graph.ParamEdge(node.constraint,node,"constraint"))
        self.queue.append(node.constraint)

   
    def visitEnsureCommonDimOp(self,node):
        self.visitUnaryUnaryOp(node)
        for slice in node.refslices:
            self.graph.addNode(slice)
            self.graph.addEdge(query_graph.ParamChoiceEdge(slice,node,"compare_slice"))
        self.queue.extend(node.refslices)
     
    def visitBroadcastOp(self,node):
        self.visitUnaryUnaryOp(node)
        for pos, slicelist in enumerate(node.refsliceslist):
            for slice in slicelist:
                self.graph.addNode(slice)
                self.graph.addEdge(query_graph.ParamChoiceListEdge(slice,node,"compare_slices",pos))
            self.queue.extend(slicelist)
       
