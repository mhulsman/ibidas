from collections import defaultdict
import random
import numpy

import manager
import create_graph
from ..utils.multi_visitor import VisitorFactory, DirectVisitorFactory, NF_ELSE
from ..utils import util
import xmlrpclib

_delay_import_(globals(),"..representor")

networkid = util.seqgen().next
class DebugVisualizer(VisitorFactory(prefixes=("node",), flags=NF_ELSE),
                     DirectVisitorFactory(prefixes=("edge",), flags=NF_ELSE), manager.Pass):

    after = set([create_graph.CreateGraph])
    @classmethod
    def run(cls, query, run_manager):
        self = cls()
        self.rand = random.randint(0,10000000)
        self.graph = run_manager.pass_results[create_graph.CreateGraph]
        self.graph.pruneGraph()
        self.server = xmlrpclib.ServerProxy("http://localhost:9000").Cytoscape
        self.network = self.server.createNetwork("network" + str(networkid()))
        
        self.unique_names = defaultdict(int)
        self.names = dict()

        self.node_name = dict()
        self.node_class = dict()
        self.node_type = dict()
        self.node_rep = dict()

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
                self.edgeKey(edge.__class__.__name__,edge)
        
        self.edgeids = self.server.createEdges(self.network,self.edge_from, self.edge_to, self.edge_type,[True] * len(self.edge_type),False)

        self.server.addNodeAttributes("name","STRING",self.node_name,False)
        self.server.addNodeAttributes("type","STRING",self.node_type,False)
        self.server.addNodeAttributes("class","STRING",self.node_class,False)
        self.server.addNodeAttributes("rep","STRING",self.node_rep,False)
        self.server.addEdgeAttributes("type","STRING",dict(zip(self.edgeids,self.edge_type)))
        self.server.addEdgeAttributes("attr","STRING",dict(zip(self.edgeids,self.edge_attr)))

        for attribute,attribute_dict in self.graph.node_attributes.iteritems():
            attribute_name_dict = {}
            if(isinstance(attribute_dict.values()[0], float)):
                xtype = "FLOATING"
                cls = float
            elif(isinstance(attribute_dict.values()[0], int)):
                xtype = "INTEGER"
                cls = int
            else:
                xtype = "STRING"
                cls = str
            for node, node_name in self.names.iteritems():
                try:
                    r = attribute_dict.get(node,"")
                    if(isinstance(r, representor.Representor)):
                        r = str(r.__class__.__name__)
                    else:
                        r = cls(r)
                    attribute_name_dict[node_name] = r
                except:
                    pass
            self.server.addNodeAttributes(attribute,xtype,attribute_name_dict,False)
            if(attribute == 'links'):
                import matplotlib.cm
                cm = discrete_color_map(attribute_name_dict.values(), matplotlib.cm.gist_rainbow)
                self.server.createDiscreteMapper('default','links', 'Node Color','#444444',cm)
            if(attribute == "time"):
                self.server.createContinuousMapper('default','time', 'Node Size',[0.0, max(attribute_dict.values())],[20.0, 20.0, 100.0, 100.0])
    
            

        self.server.setNodeLabel(self.network, "name", "","default")
        self.server.setDiscreteNodeShapeMapper(self.network, 'default',
                'type', 'diamond', {'else':'ellipse', 'unaryop':'octagon', 'rep':'round_rect'}, True)
        self.server.setEdgeTargetArrowRule(self.network,"type","Arrow",["ParamListEdge","ParamChoiceListEdge"],["T","T"])
        self.server.setEdgeLineStyleRule(self.network,"type","SOLID",["ParamChoiceEdge","ParamChoiceListEdge",'SQLResultEdge'],["DOT","DOT","SINEWAVE"])
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
        if(node.__class__.__name__[-2:] == "Op"):
            self.node_name[name] = node.__class__.__name__[:-2]
        else:
            self.node_name[name] = node.__class__.__name__
        self.node_class[name] = node.__class__.__name__
        self.node_type[name] = "else"
        self.node_rep[name] = str(node)
        return name
    
    def nodeUnaryOp(self,node):
        name =self.nodeelse(node)
        self.node_type[name] = "unaryop"

    def edgeelse(self,edge):
        self.edge_from.append(self.names[edge.source])
        self.edge_to.append(self.names[edge.target])
        self.edge_type.append(edge.__class__.__name__)
        self.edge_attr.append(str(edge))




def discrete_color_map(elems, colormap):
    import matplotlib.colors
    cconv = matplotlib.colors.ColorConverter()
    z = set(elems)
    res = {}

    steps = numpy.linspace(0.0, 1.0, len(z))
    colors = colormap(steps)
    for elem, colorrow in zip(z, colors):
        res[elem] = matplotlib.colors.rgb2hex(cconv.to_rgb(colorrow))

    return res


