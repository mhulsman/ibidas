from collections import defaultdict
from ibidas.utils import util

class Graph(object):
    def __init__(self):
        self.root = None
        self.nodes = set()
        self.node_attributes = defaultdict(dict)
        self.na = self.node_attributes
        self.edge_source = defaultdict(set)
        self.edge_target = defaultdict(set)

    def checkGraph(self):
        for key,value in self.node_attributes.iteritems():
            assert set(value.keys()) <= self.nodes, "Node attribute nodes not subset of graph ndoes"

    def setRoot(self, root):
        self.root = root

    def addNode(self,node):
        assert isinstance(node,Node), "Cannot add non-node to graph"
        self.nodes.add(node)

    def addEdge(self,edge):
        assert edge.source in self.nodes, "Unknown edge source"
        assert edge.target in self.nodes, "Unknown edge target"
        self.edge_source[edge.source].add(edge)
        self.edge_target[edge.target].add(edge)

    def dropEdge(self, edge):
        self.edge_source[edge.source].discard(edge)
        self.edge_target[edge.target].discard(edge)
       

    def getSources(self):
        targets = set()
        for key,value in self.edge_target.iteritems():
            if value:
                targets.add(key)
        return self.nodes - targets

    def pruneGraph(self):
        visited_nodes = set()
        visited_edges = set()
        node_queue = [self.root]
        while(node_queue):
            node = node_queue.pop()
            if(node in visited_nodes):
                continue
            visited_nodes.add(node)
            for edge in self.edge_target[node]:
                visited_edges.add(edge)
                node_queue.append(edge.source)
        
        self.nodes = self.nodes & visited_nodes
        for key, value in list(self.edge_source.iteritems()):
            r = value & visited_edges
            if not r:
                del self.edge_source[key]
            else:
                self.edge_source[key] = r
        
        for key, value in list(self.edge_target.iteritems()):
            r = value & visited_edges
            if not r:
                del self.edge_target[key]
            else:
                self.edge_target[key] = r

class Node(object):
    pass


class Edge(object):
    __slots__ = ["source","target","type","subtype","attr"]

    def __init__(self,source,target,type=None,subtype=None,attr=None):
        self.source = source
        self.target = target
        self.type = type
        self.subtype = subtype
        self.attr = attr

class ParamEdge(Edge):
    __slots__ = ["name"]
    def __init__(self, source, target, name):
        self.name = name
        Edge.__init__(self, source, target)

    def __str__(self):
        return self.name

class ParamListEdge(ParamEdge):
    __slots__ = ["pos"]
    
    def __init__(self, source, target, name, pos):
        self.pos = pos
        ParamEdge.__init__(self, source, target, name)

    def __str__(self):
        return str(self.pos) + ":" + self.name 

class ParamChoiceEdge(ParamEdge):
    __slots__ = []

class ParamChoiceListEdge(ParamListEdge):
    __slots__ = []
