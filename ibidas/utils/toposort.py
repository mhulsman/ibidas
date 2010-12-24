import copy

class CycleError(Exception):
    pass

class StableTopoSortGraph(object):
    def __init__(self):
        self.nodes = []
        self.parents = {}
        self.ancestors = {}

    def addNodeIfNotExist(self,node):
        if(node in self.parents):
            return False
        else:
            self.nodes.append(node)
            self.parents[node] = set()
            self.ancestors[node] = set()
            return True

    def mergeNodes(self,nodeid1,nodeid2):
        if(nodeid1 in self.ancestors[nodeid2] or nodeid2 in self.ancestors[nodeid1]):
            raise CycleError, "Merger introduces cycle!"

        del self.nodes[self.nodes.index(nodeid1)]
        self.parents[nodeid2].update(self.parents[nodeid1])
        self.ancestors[nodeid2].update(self.ancestors[nodeid1])
        del self.parents[nodeid1]
        del self.ancestors[nodeid1]
        for node in self.nodes:
            if nodeid1 in self.parents[node]:
                self.parents[node].discard(nodeid1)
                self.parents[node].add(nodeid2)
            if nodeid1 in self.ancestors[node]:
                self.ancestors[node].discard(nodeid1)
                self.ancestors[node].add(nodeid2)

    def addEdge(self,before,after):
        assert before in self.parents, "Source node not known"
        assert after in self.parents, "Target node not known"

        if after in self.ancestors[before] or after == before:
            raise CycleError, "Edge introduces cycle!"

        self.parents[after].add(before)
        self.ancestors[after].add(before)
        self.ancestors[after].update(self.ancestors[before])

    def getParents(self,node):
        return self.parents[node]

    def getAncestors(self,node):
        return self.ancestors[node]

    def getDescendants(self,node):
        return [n for n in nodes if node in self.ancestors[n]]

    def copy(self):
        return copy.deepcopy(self)

    def __iter__(self):
        return StableTopoSortIter(self)
    
    def elem(self,n,after=None,exclude=None):
        i = self.__iter__()
        if(not after is None):
            for node in i:
                if(node == after):
                    break
            else:
                raise Indexerror, "Not found"

        pos = -1
        for node in i:
            if(exclude is None or not node in exclude):
                pos += 1
            if(pos == n):
                break
        else:
            raise IndexError, "Not found"
        return node


class StableTopoSortIter(object):
    def __init__(self,stable_topo_sort_graph):
        self.sts_graph = stable_topo_sort_graph
        self.visited = set()

    def __iter__(self):
        return self

    def next(self):
        nodes = self.sts_graph.nodes
        visit_node = None
        
        if(len(self.visited) == len(self.sts_graph.nodes)):
            raise StopIteration
        
        for node in nodes:
            if(node in self.visited):
                continue
            parents = self.sts_graph.parents[node]
            if(parents.issubset(self.visited)):
                visit_node = node
                break
        if(visit_node is None):
            raise RuntimeError, "No viable next element during toposort, cycle?"
        self.visited.add(visit_node)
        return visit_node

class StableTopoSortCopyIter(StableTopoSortIter):
    def __init__(self,stable_topo_sort_graph):
        self.sts_graph = stable_topo_sort_graph.copy()
        self.visited = set()

    def __iter__(self):
        return self
