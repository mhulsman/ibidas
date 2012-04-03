import numpy
import xmlrpclib
from ibidas.utils import util,cutils

_delay_import_(globals(),"ibidas","*")
_delay_import_(globals(),"..itypes","rtypes","dimpaths","dimensions")


@util.memoized
def cyconnect(url=None):
    import xmlrpclib
    if url is None:
        url = "http://localhost:9000"
    return xmlrpclib.ServerProxy("http://localhost:9000").Cytoscape


def nan_to_zero(val):
    if numpy.isnan(val):
        return 0.
    else:
        return val


class CyNetwork(object):
    def __init__(self, name=None, url=None, exists=False):
        self._server = cyconnect(url)

        networkid = None
        if exists:
            if name is None:
                assert self._server.hasCurrentNetwork(), "No current network found"
                self._networkid = self._server.getCurrentNetworkIndex()
            else:
                nlist = self._server.getNetworkList()
                nlist = [elem for elem in nlist if elem['networktitle'] == name]
                assert len(nlist) > 1, "Multiple networks with same name. Maybe select on current network (no name)?"
                self._networkid = nlist[0]['networkID']
        else:                
            if name is None:
                name = "Network"
            self._networkid = self._server.createNetwork(name)


    def _getServ(self):
        return self._server
    Direct = property(fget=_getServ)

    def _getId(self):
        return self._networkid
    Id = property(fget=_getId)

    def AddNodes(self, rep, primary=0):
        assert isinstance(rep, representor.Representor), "Rep should be a representor object"
        rep = rep.Copy()

        primary = rep.Get(primary)
        attributes = rep.Without(primary)
        attributes = attributes.Level(1)
        
        #package data if needed
        primary = primary.Level(1).Cast("bytes")
       
        #some checks
        assert len(primary.Slices) == 1, "Node identifiers should consist of only one slice"
        dims = set([primary.Slices[0].dims]) | set([aslice.dims for aslice in attributes.Slices])
        assert len(dims) == 1, "Multiple root dimensions not allowed"

        nodeids = primary.ToPython()
        self._server.createNodes(self._networkid, nodeids)
        
        for pos, slice in enumerate(attributes.Slices):
            attribute = attributes.Get(pos)
            xtype, attribute = self._prepareAttribute(attribute)

            attrmap = dict(Combine(primary, attribute).Tuple().ToPython())
            self._server.addNodeAttributes(slice.name, xtype,attrmap,False)
            
    def AddEdges(self, rep, left=0, right=1, itype=None, directed=True, addNodes=True):
        assert isinstance(rep, representor.Representor), "Rep should be a representor object"
        if itype is None:
            itype = Rep("Interaction")

        directed = Rep(directed)

        rep = rep.Copy()

        primary = rep.Get(left, right, itype, directed)
        attributes = rep.Without(primary)
        attributes = attributes.Level(1)
        
        #package data if needed
        primary = primary.Level(1).To(0,1,2,Do=_.Cast("bytes"))
        
        #some checks
        assert len(primary.Slices) == 4, "Edge identifiers should consist of four slices"
        dims = set([pslice.dims for pslice in primary.Slices]) | set([aslice.dims for aslice in attributes.Slices])
        assert len(dims) == 1, "Multiple root dimensions not allowed"

        if addNodes:
            nodes = (primary.Get(0).Set() | primary.Get(1).Set()).Elem()
            self.AddNodes(nodes)

        edgeids = primary.Array(0).ToPython()
        cedgeids = self._server.createEdges(self._networkid, edgeids[0], edgeids[1], edgeids[2], edgeids[3], False)
        
        for pos, slice in enumerate(attributes.Slices):
            attribute = attributes.Get(pos)
            xtype, attribute = self._prepareAttribute(attribute)
            attrmap = dict(zip(cedgeids, attribute.ToPython()))
            self._server.addEdgeAttributes(slice.name, xtype, attrmap)


    def _prepareAttribute(self, attribute):
        assert len(attribute.Slices) == 1, "Attribtue slice nr should be 1"
        slice = attribute.Slices[0]
        if(isinstance(slice.type, rtypes.TypeBool) and not slice.type.has_missing):
            xtype = "BOOLEAN"
        elif(isinstance(slice.type, rtypes.TypeInteger) and not slice.type.has_missing):
            xtype = "INTEGER"
        elif(isinstance(slice.type, rtypes.TypeReal64) and not slice.type.has_missing):
            xtype = "FLOATING"
            attribute = attribute.Each(nan_to_zero,dtype="real64")
        else:
            xtype = "STRING"
            attribute = attribute.Cast("bytes")
        
        return (xtype, attribute)

    def GetLayoutNames(self):
        return Rep(self._server.getLayoutNames())

    def PerformLayout(self, name=None):
        if name is None:
            self._server.performDefaultLayout()
        else:
            self._server.performLayout(self._networkid,name)

    def SetNodeLabel(self, name=None):
        if name is None:
            name = "canonicalName"
        self._server.setNodeLabel(self._networkid, name, "", "default")

    def Nodes(self): 
        return Rep(self._server.getNodes(self._networkid),dtype="[nodes:*]<bytes")

    def Edges(self):
        return Rep([parse(edgeid) for edgeid in self._server.getEdges(self._networkid)],dtype="[edges:*]<(edgeid=bytes, left=bytes,right=bytes,type=bytes)")
    
    def SelNodes(self): 
        return Rep(self._server.getSelectedNodes(self._networkid),dtype="[selnodes:*]<bytes")

    def SelEdges(self):
        return Rep([parse(edgeid) for edgeid in self._server.getSelectedEdges(self._networkid)],dtype="[seledges:*]<(edgeid=bytes, left=bytes,right=bytes,type=bytes)")

    def NodeAttributeInfo(self):
        attr_names = self._server.getNodeAttributeNames()
        if 'hiddenLabel' in attr_names:
            del attr_names[attr_names.index('hiddenLabel')]
        attr_types = [self._server.getNodeAttributeType(attr_name) for attr_name in attr_names]
        return Rep((attr_names, attr_types), dtype="(name=[node_attributes:*]<bytes,type=[node_attributes:*]<bytes)")

    def EdgeAttributeInfo(self):
        attr_names = self._server.getEdgeAttributeNames()
        attr_types = [self._server.getEdgeAttributeType(attr_name) for attr_name in attr_names]
        return Rep((attr_names, attr_types), dtype="(name=[edge_attributes:*]<bytes,type=[edge_attributes:*]<bytes)")

    def NodeAttributes(self, nodes=None, attributes=None):
        if(isinstance(nodes,representor.Representor)):
            nodes = nodes.ToPython()
        if(isinstance(attributes, representor.Representor)):
            attributes = attributes.ToPython()

        if nodes is None:
            nodes = self._server.getNodes(self._networkid)      

        xattributes = self.NodeAttributeInfo()
        if attributes is None:
            attributes = xattributes
        else:   
            attributes = xattributes[_.name |In| attributes]

        columns = []
        fieldnames = []
        fieldtypes = []
        dimpath=dimpaths.DimPath(dimensions.Dim(shape=UNDEFINED,has_missing=False,dependent=(),name="node_attributes"))
        for attribute, atype in attributes.Tuple()():
            atype = self._prepareType(atype)
            has_attribute = util.darray(self._server.nodesHaveAttribute(attribute,nodes) ,bool)
            if False in has_attribute:
                atype.has_missing=True
                res = util.darray([Missing] * len(nodes))
                fnodes = list(util.darray(nodes)[has_attribute])
                res[has_attribute] = self._server.getNodesAttributes(attribute,fnodes)
            else:
                res = self._server.getNodesAttributes(attribute, nodes)
            columns.append(res)
            fieldnames.append(util.valid_name(attribute))
            fieldtypes.append(rtypes.TypeArray(dims=dimpath,subtypes=(atype,)))
            
        dtype = rtypes.TypeTuple(fieldnames=tuple(fieldnames),subtypes=tuple(fieldtypes))
        return Rep(columns,dtype=dtype)

    def EdgeAttributes(self, edges=None, attributes=None):
        if(isinstance(edges,representor.Representor)):
            edges = edges.ToPython()
        if(isinstance(attributes, representor.Representor)):
            attributes = attributes.ToPython()

        if edges is None:
            edges = self._server.getEdges(self._networkid)      

        xattributes = self.EdgeAttributeInfo()
        if attributes is None:
            attributes = xattributes
        else:   
            attributes = xattributes[_.name |In| attributes]

        columns = []
        fieldnames = []
        fieldtypes = []
        dimpath=dimpaths.DimPath(dimensions.Dim(shape=UNDEFINED,has_missing=False,dependent=(),name="edge_attributes"))
        for attribute, atype in attributes.Tuple()():
            atype = self._prepareType(atype)
            has_attribute = util.darray(self._server.edgesHaveAttribute(attribute,edges) ,bool)
            if False in has_attribute:
                atype.has_missing=True
                res = util.darray([Missing] * len(edges))
                fedges = list(util.darray(edges)[has_attribute])
                res[has_attribute] = self._server.getEdgesAttributes(attribute,fedges)
            else:
                res = self._server.getEdgesAttributes(attribute, edges)
            columns.append(res)
            fieldnames.append(util.valid_name(attribute))
            fieldtypes.append(rtypes.TypeArray(dims=dimpath,subtypes=(atype,)))
            
        dtype = rtypes.TypeTuple(fieldnames=tuple(fieldnames),subtypes=tuple(fieldtypes))
        return Rep(columns,dtype=dtype)


    def _prepareType(self, cytype):
        if(cytype == "FLOATING"):
            return rtypes.TypeReal64()
        elif(cytype == "INTEGER"):
            return rtypes.TypePlatformInt()
        elif(cytype == "BOOLEAN"):
            return rtypes.TypeBoolean()
        elif(cytype == "STRING"):
            dim = dimensions.Dim(shape=UNDEFINED, has_missing=False, dependent=(True,))
            return rtypes.TypeBytes(dims=dimpaths.DimPath(dim))
        else:
            return rtypes.unknown

    def _getNodeCount(self):
        return self._server.countNodes(self._networkid)
    NodeCount=property(fget=_getNodeCount)

    def _getEdgeCount(self):
        return self._server.countEdges(self._networkid)
    EdgeCount=property(fget=_getEdgeCount)

    def _getName(self):
        return self._server.getNetworkTitle(self._networkid)
    Name = property(fget=_getName)

    def __repr__(self):
        return "<CyNetwork " + str(self.Name) + ": " + str(self.NodeCount) + " nodes, " + str(self.EdgeCount) + " edges>"
      
    def CreateDiscreteColorMap(self, elems, colormap=None):
        import matplotlib.cm
        import matplotlib.colors
        if colormap is None:
            colormap = matplotlib.cm.gist_rainbow
        cconv = matplotlib.colors.ColorConverter()
        z = set(elems)
        res = {}

        steps = numpy.linspace(0.0, 1.0, len(z))
        colors = colormap(steps)
        for elem, colorrow in zip(z, colors):
            res[elem] = matplotlib.colors.rgb2hex(cconv.to_rgb(colorrow))

        return res

def parse(edgeid):
    split = edgeid.split(' (')
    left = split[0]
    rest = " (".join(split[1:])
    split = rest.split(') ')
    right = split[-1]
    type = " )".join(split[:-1])
    return (edgeid, left, right, type)

