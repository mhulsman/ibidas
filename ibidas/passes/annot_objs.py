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
    def getDims(self, graph):
        sops = self.getNodesByClass(ops.SelectOp)
        assert len(sops) == 2, "Unexpected number of select operations"
        lsop,rsop = sops
        if(lsop.index == 1):
            lsop,rsop = rsop,lsop
        
        lfilters = [edge.target for edge in graph.edge_source[lsop]]
        rfilters = [edge.target for edge in graph.edge_source[rsop]]

        assert all([isinstance(lfilter,ops.FilterOp) for lfilter in lfilters]), "Non-filter target of select in match"
        assert all([isinstance(rfilter,ops.FilterOp) for rfilter in rfilters]), "Non-filter target of select in match"

        ldims = set([lfilter.dims for lfilter in lfilters])
        rdims = set([rfilter.dims for rfilter in rfilters])
        return ldims, rdims

    def getType(self):
        sops = self.getNodeByClass(ops.EquiJoinIndexOp)
        return sops.jointype
        
    
    def getComparisonEdges(self,graph):
        jindex = self.getNodeByClass(ops.EquiJoinIndexOp)
        ledge = self.sourceDataEdge(graph.getDataEdge(jindex,pos=0),graph)
        redge = self.sourceDataEdge(graph.getDataEdge(jindex,pos=1),graph)
        return (ledge,redge)

    def getInfo(self,graph):
        sops = self.getNodesByClass(ops.SelectOp)
        assert len(sops) == 2, "Unexpected number of select operations"
        lsop,rsop = sops
        if(lsop.index == 1):
            lsop,rsop = rsop,lsop
        
        lfilters = [edge.target for edge in graph.edge_source[lsop]]
        rfilters = [edge.target for edge in graph.edge_source[rsop]]
        ledges = [self.sourceDataEdge(graph.getDataEdge(lfilter,0),graph) for lfilter in lfilters]
        redges = [self.sourceDataEdge(graph.getDataEdge(rfilter,0),graph) for rfilter in rfilters]

        olnodes = [self.targetDataNode(lfilter,graph) for lfilter in lfilters]
        ornodes = [self.targetDataNode(rfilter,graph) for rfilter in rfilters]

        return (ledges,redges,olnodes,ornodes)

       
