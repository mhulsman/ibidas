from collections import defaultdict
from ibidas.utils import util
import copy

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

    def copyNode(self, node):
        nnode = copy.copy(node)
        self.nodes.add(nnode)
        if node in self.edge_source:
            for edge in list(self.edge_source[node]):
                self.dropEdge(edge)
                edge.source = nnode
                self.addEdge(edge)
            del self.edge_source[node]
        if node in self.edge_target:
            for edge in list(self.edge_target[node]):
                self.dropEdge(edge)
                edge.target = nnode
                self.addEdge(edge)
            del self.edge_target[node]
        self.nodes.discard(node)
        for key, value in self.node_attributes.iteritems():
            if node in value:
                value[nnode] = value[node]
                del value[node]
        return nnode

    def dropNode(self, node):
        if node in self.edge_source:
            for edge in list(self.edge_source[node]):
                self.dropEdge(edge)
            del self.edge_source[node]
        if node in self.edge_target:
            for edge in list(self.edge_target[node]):
                self.dropEdge(edge)
            del self.edge_target[node]
        self.nodes.discard(node)
        for key, value in self.node_attributes.iteritems():
            if node in value:
                del value[node]

    def dropEdge(self, edge):
        self.edge_source[edge.source].discard(edge)
        self.edge_target[edge.target].discard(edge)
   
    def getDataEdge(self, node, pos=0, name="slice"):
        for edge in self.edge_target[node]:
            if(isinstance(edge,ParamEdge) and edge.name == name):
                assert pos == 0, "Cannot have pos larger than 0"
                return edge
            elif(isinstance(edge,ParamListEdge) and edge.name == name + "s" and edge.pos == pos):
                return edge
        raise RuntimeError, "Edge with pos: " + str(pos) + " not found for node: " + str(node) + ":" + str(node.__class__.__name__)

    def getEdge(self, source, target):
        res = self.getEdges(source, target)
        assert len(res) == 1, "More than one edge found"
        return res[0]
    
    def getEdges(self, source, target):
        edges = self.edge_source[source]
        return [edge for edge in edges if edge.target == target]

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


    def walkUseContig(self, node, include_first=True):
        return WalkContigIter(self, node, True, include_first)

    def walkSourceContig(self, node, include_first=True):
        return WalkContigIter(self, node, False, include_first)
    
    def remove_unaryop(self, node):
        source = self.getDataEdge(node).source
        target_edges = list(self.edge_source[node])
        self.dropNode(node)
        for target_edge in target_edges:
            assert isinstance(target_edge,ParamEdge), "Unknown edge type encountered"
            target_edge.source = source
            self.addEdge(target_edge)
        return source            


class WalkContigIter(object):
    def __init__(self, graph, node, walk_users, include_first=True):
        self.node = node
        self.graph = graph
        self.walk_users = walk_users
        if not include_first:
            self.next()

    def __iter__(self):
        return self

    def next(self):
        node = self.node
        if not node is None:
            if self.walk_users:
                edges = self.graph.edge_source[self.node]
                if len(edges) > 1:
                    self.node = None
                else:
                    edge = list(edges)[0]
                    self.node = edge.target
            else: #walk_source
                edges = self.graph.edge_target[self.node]
                if len(edges) > 1:
                    self.node = None
                else:
                    edge = list(edges)[0]
                    self.node = edge.source
            return node
        else:
            raise StopIteration
    

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
