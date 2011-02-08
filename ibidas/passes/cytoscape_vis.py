from collections import defaultdict
import random

import manager
import create_graph
from ..utils.multi_visitor import VisitorFactory, DirectVisitorFactory, NF_ELSE
from ..utils import util
import xmlrpclib

networkid = util.seqgen().next
class DebugVisualizer(VisitorFactory(prefixes=("node",), flags=NF_ELSE),
                     DirectVisitorFactory(prefixes=("edge",), flags=NF_ELSE), manager.Pass):

    after = set([create_graph.CreateGraph])
    @classmethod
    def run(cls, query, run_manager):
        self = cls()
        self.rand = random.randint(0,10000000)
        self.graph = run_manager.pass_results[create_graph.CreateGraph]
        self.server = xmlrpclib.ServerProxy("http://localhost:9000").Cytoscape
        self.network = self.server.createNetwork("network" + str(networkid()))
        
        self.unique_names = defaultdict(int)
        self.names = dict()

        self.node_name = dict()
        self.node_class = dict()
        self.node_type = dict()

        self.edge_from = []
        self.edge_to = []
        self.edge_type = []
        self.edge_attr = []

        for node in self.graph.nodes:
            self.node(node)

        self.server.createNodes(self.network, self.names.values())
        for source,edges in self.graph.edge_source.iteritems():
            for edge in edges:
                assert edge.source is source, "Source in edge and index not equal"
                self.edgeKey(edge.type,edge)
        
        self.edgeids = self.server.createEdges(self.network,self.edge_from, self.edge_to, self.edge_type,[True] * len(self.edge_type),False)

        self.server.addNodeAttributes("name","STRING",self.node_name,False)
        self.server.addNodeAttributes("type","STRING",self.node_type,False)
        self.server.addNodeAttributes("class","STRING",self.node_class,False)
        self.server.addEdgeAttributes("type","STRING",dict(zip(self.edgeids,self.edge_type)))
        self.server.addEdgeAttributes("attr","STRING",dict(zip(self.edgeids,self.edge_attr)))

        for attribute,attribute_dict in self.graph.node_attributes.iteritems():
            attribute_name_dict = dict([(node_name, str(attribute_dict.get(node,""))) for node,node_name in self.names.iteritems()])
            self.server.addNodeAttributes(attribute,"STRING",attribute_name_dict,False)
        
        self.server.setNodeLabel(self.network, "name", "","default")
        self.server.setDiscreteNodeShapeMapper(self.network, 'default',
                'type', 'diamond', {'else':'ellipse', 'unaryop':'octagon', 'rep':'round_rect'}, True)
        self.server.setEdgeTargetArrowRule(self.network,"type","Arrow",["paramlist","paramchoicelist"],["T","T"])
        self.server.setEdgeLineStyleRule(self.network,"type","SOLID",["paramchoice","paramchoicelist"],["DOT","DOT"])
        self.server.performLayout(self.network, "hierarchical")
   
    def createUniqueName(self,name):
        counter = self.unique_names[name]
        self.unique_names[name] += 1
        counter -= 1

        if(counter < 0):
            return name + str(self.rand)
        else:
            return name + str(self.rand) + "_" + str(counter)

    def nodeelse(self,node):
        name = self.createUniqueName(node.__class__.__name__)
        self.names[node] = name
        self.node_name[name] = node.__class__.__name__
        self.node_class[name] = node.__class__.__name__
        self.node_type[name] = "else"
        return name
    
    def nodeUnaryOp(self,node):
        name =self.nodeelse(node)
        self.node_type[name] = "unaryop"
        self.node_name[name] = node.__class__.__name__

    def edgeelse(self,edge):
        self.edge_from.append(self.names[edge.source])
        self.edge_to.append(self.names[edge.target])
        self.edge_type.append(edge.type)
        self.edge_attr.append(str(edge.attr))


