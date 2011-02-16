from ibidas.utils import util
_delay_import_(globals(),"..ops")
_delay_import_(globals(),"..query_graph")

class Expression(object):
    __slots__ = ["etype","eobj","in_slices","out_slices","all_slices"]
    def __init__(self, etype, eobj):
        self.etype = etype
        self.eobj = eobj
        self.in_slices = set()
        self.out_slices = set()
        self.all_slices = set()

    def addInSlice(self, in_slice):
        self.in_slices.add(in_slice)

    def addOutSlice(self, out_slice):
        self.out_slices.add(out_slice)

    def addAllSlice(self, between_slice):
        self.all_slices.add(between_slice)

    def sourceDataEdge(self, edge, graph):
        links = graph.node_attributes['links']
        while(links[edge.source] == self):
            edge = graph.getDataEdge(edge.source)
        return edge
    
    def targetDataNode(self, node, graph):
        links = graph.node_attributes['links']
        
        edges = [edge for edge in graph.edge_source[node] if isinstance(edge,query_graph.ParamEdge)]
        edge = edges[0]
        if len(edges) > 1:
            assert all([links[edge.target] != self for edge in edges]), "Cannot decide on target data node"

        while(links[edge.target] == self):
            edges = [edge for edge in graph.edge_source[edge.target] if isinstance(edge,query_graph.ParamEdge)]
            edge = edges[0]
            if len(edges) > 1:
                assert all([links[edge.target] != self for edge in edges]), "Cannot decide on target data node"

        return edge.source
   
    def getNodesByClass(self, cls):
        return [node for node in self.all_slices if node.__class__ == cls]

    def getNodeByClass(self, cls):
        res = self.getNodesByClass(cls)
        assert len(res) ==1, "No nodes or multiple nodes with the same class found"
        return res[0]

    def __str__(self):
        return str(self.etype)

class BinFuncElemExpression(Expression):
    def getOp(self):
        return (self.getOutSlice().sig.name, self.eobj.__class__.__name__)

    def getOutSlice(self):
        return self.getNodeByClass(ops.BinFuncElemOp)


    def getLeftRightInEdges(self, graph):
        c = self.getNodeByClass(ops.BinFuncElemOp)
        e1 = self.sourceDataEdge(graph.getDataEdge(c,0), graph)
        e2 = self.sourceDataEdge(graph.getDataEdge(c,1), graph)
        return (e1,e2)

    def getOutEdges(self, graph):
        c = self.getNodeByClass(ops.BinFuncElemOp)
        out_edges = graph.edge_source[c]
        return out_edges

class FilterExpression(Expression):
    def getDims(self):
        filters = self.getNodesByClass(ops.FilterOp)
        return set([filter.dims for filter in filters])
        
    
    def getInfo(self, graph):
        filters = self.getNodesByClass(ops.FilterOp)
        cedge = self.sourceDataEdge(graph.getDataEdge(filters[0],name="constraint"), graph)
        fedges = [self.sourceDataEdge(graph.getDataEdge(f), graph) for f in filters]
        onodes = [self.targetDataNode(f, graph) for f in filters]
        return (fedges,cedge, onodes)


class MatchExpression(Expression):
    def getUsedSources(self):
        sops = self.getNodesByClass(ops.SelectOp)
        res = [False,False]
        for sop in sops:
            res[sop.index] = True
        return res
       
    def getSop(self, index):
        sops = self.getNodesByClass(ops.SelectOp)
        for sop in sops:
            if sop.index == index:
                return sop
        else:
            raise RuntimeError, "Could not find sop"
    
    def getDim(self,index, graph):
        sop = self.getSop(index)
        filters = [edge.target for edge in graph.edge_source[sop]]
        assert all([isinstance(filter,ops.FilterOp) for filter in filters]), "Non-filter target of select in match"
        dims = set([filter.dims for filter in filters])
        return dims

    def getType(self):
        sops = self.getNodeByClass(ops.EquiJoinIndexOp)
        return sops.jointype
    
    def getComparisonEdges(self,graph):
        jindex = self.getNodeByClass(ops.EquiJoinIndexOp)
        ledge = self.sourceDataEdge(graph.getDataEdge(jindex,pos=0),graph)
        redge = self.sourceDataEdge(graph.getDataEdge(jindex,pos=1),graph)
        return (ledge,redge)


    def getInfo(self, index, graph):
        sop = self.getSop(index)
        filters = [edge.target for edge in graph.edge_source[sop]]
        edges = [self.sourceDataEdge(graph.getDataEdge(filter,0),graph) for filter in filters]
        onodes = [self.targetDataNode(filter,graph) for filter in filters]

        return (edges,onodes)

       
