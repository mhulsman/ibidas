from collections import defaultdict

class Graph(object):
    def __init__(self):
        self.root = None
        self.nodes = set()
        self.node_attributes = defaultdict(dict)
        self.na = self.node_attributes
        self.edge_source = defaultdict(set)
        self.edge_target = defaultdict(set)

    def setRoot(self, root):
        self.root = root

    def addNode(self,node):
        assert isinstance(node,Node), "Cannot add non-node to graph"
        self.nodes.add(node)

    def addEdge(self,edge):
        self.edge_source[edge.source].add(edge)
        self.edge_target[edge.target].add(edge)


class Node(object):
    __slots__ = []


class Edge(object):
    __slots__ = ["source","target","type","subtype","attr"]

    def __init__(self,source,target,type=None,subtype=None,attr=None):
        self.source = source
        self.target = target
        self.type = type
        self.subtype = subtype
        self.attr = attr
