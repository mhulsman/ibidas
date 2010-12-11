import operator
from itertools import chain
from collections import defaultdict
import numpy

import wrapper
from ..constants import *
from ..utils.multi_visitor import VisitorFactory, DirectVisitorFactory, NF_ERROR, NF_ELSE
from ..itypes import rtypes, dimpaths
from ..passes import manager, create_graph, serialize_exec
from .. import slices

from ..utils.multi_visitor import VisitorFactory, DirectVisitorFactory, NF_ELSE

_delay_import_(globals(),"..representor")
_delay_import_(globals(),"..utils","util","nestutils","cutils","nested_array")
_delay_import_(globals(),"..itypes","detector","type_attribute_freeze","convertors")
_delay_import_(globals(),"..repops_slice")
_delay_import_(globals(),"..repops_dim")


class PyRepresentor(wrapper.SourceRepresentor):
    pass

def rep(data=None, dtype=None, unpack=True, name=None):#{{{
    if(not dtype is None):
        if(isinstance(dtype,str)):
            dtype = rtypes.createType(dtype)
    else:
        det = detector.Detector()
        det.process(data)
        dtype = det.getType()
    
    if(name is None):
        name = "data"

    data_slice = slices.DataSlice(data,name=name,rtype=dtype)
    data_slice = slices.ensure_normal_or_frozen(data_slice)
    
    while(unpack and data_slice.type.__class__ is rtypes.TypeArray):
        data_slice = slices.ensure_normal_or_frozen(slices.UnpackArraySlice(data_slice))

    res = wrapper.SourceRepresentor()
    res.initialize((data_slice,)) 

    if(unpack and isinstance(res.getType(), rtypes.TypeTuple)):
        res = repops_slice.UnpackTuple(res)
    
    return res#}}}


class ResultSlice(slices.DataSlice):
    __slots__ = ["source"]
    def __init__(self, data=None, name=None, rtype=rtypes.unknown, dims=dimpaths.DimPath(), bookmarks=set()):
        self.name = name
        self.type = rtype
        self.dims = dims
        self.data = data
        self.bookmarks = bookmarks

    @classmethod
    def from_slice(cls,data,slice):
        return cls(data,slice.name, slice.type,slice.dims,slice.bookmarks)

    def setSource(self,source):
        self.source = source

    def copy(self, data=None):
        if(data is None):
            data = self.data

        return ResultSlice(data, self.name, self.type, self.dims, self.bookmarks)
       

    def __repr__(self):
        return  "\nName: " + str(self.name) + ", Type:   " + str(self.type) + ", Dims: " + str(self.dims) + \
                "\n" + str(self.data) 


class PyExec(VisitorFactory(prefixes=("visit",), flags=NF_ELSE), 
             DirectVisitorFactory(prefixes=("cast",),flags=NF_ERROR), manager.Pass):

    after = set([create_graph.CreateGraph, serialize_exec.SerializeExec])

    @classmethod
    def run(cls, query, run_manager,debug_mode=False):
        self = cls()
        self.graph = run_manager.pass_results[create_graph.CreateGraph]
        commands = run_manager.pass_results[serialize_exec.SerializeExec]

        arguments = []
        use_counters = []
        param_idxs = self.graph.na["param_idxs"]
        param_usecount = self.graph.na["param_usecount"]

        for command in commands:
            command_id = len(arguments)
            param_args,param_kwds = param_idxs[command]
           
            #create kwds/args 
            if(param_kwds):
                param_kwds = param_kwds.copy()
                for param_name,idx in param_kwds.iteritems():
                    if(use_counters[idx] == 1):
                        use_counters[idx] = 0
                        param_kwds[param_name] = arguments[idx]
                        arguments[idx] = None
                    else:
                        param_kwds[param_name] = arguments[idx].copy()
                        use_counters[idx] -= 1
            if(param_args):
                param_args = list(param_args)
                for pos in xrange(len(param_args)):
                    idx = param_args[pos]
                    if(use_counters[idx] == 1):
                        use_counters[idx] = 0
                        param_args[pos] = arguments[idx]
                        arguments[idx] = None
                    else:
                        param_args[pos] = arguments[idx].copy()
                        use_counters[idx] -= 1
               

            res = self.visit(command, *param_args, **param_kwds)
            if(debug_mode):
                self.graph.na["output"][command] = str(res)
            arguments.append(res)
            use_counters.append(param_usecount[command])
        return res

    def visitelse(self,node, *args, **kwargs):
        raise RuntimeError, "Unknown node type encountered: " + node.__class__.__name__ + " with " + str(args) + " and " + str(kwargs)


    def visitDataSlice(self,node):
        ndata = nested_array.NestedArray(node.data,node.type)
        return ResultSlice.from_slice(ndata,node)

    def visitConvertSlice(self,node,param0):
        ndata = param0.data.mapseq(node.convertor.convert,
                                node.type,res_type=node.type)
        param0.data = ndata
        return param0
   
    def visitCastSlice(self,node,param0):
        ndata = self.cast(node.cast_name,node,param0)
        param0.data = ndata
        param0.type = node.type
        return param0

    def visitChangeNameSlice(self,node,param0):
        param0.name = node.name
        return param0

    def visitChangeDimPathSlice(self,node,param0):
        param0.dims = node.dims
        return param0

    def visitChangeBookmarkSlice(self,node,param0):
        param0.bookmarks = node.bookmarks
        return param0

    def visitDetectTypeSlice(self,node,param0):
        det = detector.Detector()
        det.setParentDimensions(node.dims)
        det.processSeq(node.param0.data.flat())
        param0.type = det.getType()
        return param0
 
    
    def visitUnpackArraySlice(self,node,param0):
        ndata=param0.data.unpack(node.unpack_dims, subtype=node.type)
        param0.data = ndata
        param0.type = node.type
        param0.dims = node.dims
        return param0

    def visitPackArraySlice(self, node, param0):
        ndata=param0.data.pack(node.type, len(node.type.dims))
        param0.data = ndata
        param0.type = node.type
        param0.dims = node.dims
        return param0

    def visitPackListSlice(self, node, param0):
        ndata=param0.data.pack(node.type, len(node.type.dims))
        ndata=ndata.map(list,res_type=node.type)
        param0.data = ndata
        param0.type = node.type
        param0.dims = node.dims
        return param0

    def visitUnpackTupleSlice(self,node,param0):
        func = operator.itemgetter(node.tuple_idx)
        ndata = param0.data.map(func,res_type=node.type)
        param0.data = ndata
        param0.type = node.type
        param0.name = node.name
        return param0
 
    def visitPackTupleSlice(self,node, *params):
        ndata = nested_array.co_mapseq(speedtuplify,[param.data for param in params],res_type=node.type)
        nparam = params[0]
        nparam.data = ndata
        nparam.type = node.type
        nparam.dims = node.dims
        return nparam
    
    def visitHArraySlice(self,node, *params):
        ndata = nested_array.co_mapseq(speedarrayify,[param.data for param in params],dtype=node.type.subtypes[0].toNumpy(), res_type=node.type)
        nparam = params[0]
        nparam.data = ndata
        nparam.type = node.type
        nparam.dims = node.dims
        return nparam

    def visitFixate(self,node,*params):
        res = []
        for cur_slice, param in zip(node._slices, params):
            ndata = param.data.getStructuredData()
            nparam = param.copy()
            nparam.data = ndata
            nparam.setSource(cur_slice)
            res.append(nparam)
        return res
    
    def castto_any(self,castname,node,param0):
        return param0.data.mapseq(lambda x:x,res_type=node.type)

    def castnumbers_numbers(self,castname,node,param0):
        return param0.data.mapseq(lambda x:x,res_type=node.type)

#util funcs
def speedtuplify(seqs):
    nseq = cutils.darray(zip(*seqs))
    return nseq

def speedarrayify(seqs,dtype):
    nseq = numpy.array(seqs,dtype).T
    return nseq

