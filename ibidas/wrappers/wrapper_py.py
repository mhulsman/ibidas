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

    def modify(self,data=None, name=None, rtype=None, dims=None, bookmarks=None):
        if(name is None):
            name = self.name
        if(rtype is None):
            rtype = self.type
        if(dims is None):
            dims = self.dims
        if(data is None):
            data = self.data
        if(bookmarks is None):
            bookmarks = self.bookmarks

        return ResultSlice(data, name, rtype, dims, bookmarks)

    def setSource(self,source):
        self.source = source

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

        arguments = numpy.zeros((len(commands),),dtype=object)
        use_counters = numpy.zeros((len(commands),),dtype=int)
        param_idxs = self.graph.na["param_idxs"]
        param_usecount = self.graph.na["param_usecount"]

        for command_id, command in enumerate(commands):
            params = param_idxs[command]
          
            param_kwds = dict()
            for name,param_ids in params.iteritems():
                param_kwds[name] = arguments[param_ids]
                use_counters[param_ids] -= 1
                
            res = self.visit(command, **param_kwds)
            if(debug_mode):
                self.graph.na["output"][command] = str(res)

            arguments[command_id] = res
            use_counters[command_id] = param_usecount[command]
            arguments[use_counters == 0] = None
        return res

    def visitelse(self,node, **kwargs):
        raise RuntimeError, "Unknown node type encountered: " + node.__class__.__name__ + " with  " + str(kwargs)


    def visitDataSlice(self,node):
        ndata = nested_array.NestedArray(node.data,node.type)
        return ResultSlice.from_slice(ndata,node)

    def visitConvertSlice(self,node,slice):
        ndata = slice.data.mapseq(node.convertor.convert,
                                node.type,res_type=node.type)
        return slice.modify(data=ndata)
   
    def visitCastSlice(self,node,slice):
        ndata = self.cast(node.cast_name,node,slice)
        return slice.modify(data=ndata,type=node.type)

    def visitChangeNameSlice(self,node,slice):
        return slice.modify(name = node.name)

    def visitChangeDimPathSlice(self,node,slice):
        return slice.modify(dims=node.dims)

    def visitChangeBookmarkSlice(self,node,slice):
        return slice.modify(bookmarks=node.bookmarks)

    def visitDetectTypeSlice(self,node,slice):
        det = detector.Detector()
        det.setParentDimensions(node.dims)
        det.processSeq(node.slice.data.flat())
        return slice.modify(rtype=det.getType())
    
    def visitUnpackArraySlice(self,node,slice):
        ndata=slice.data.unpack(node.unpack_dims, subtype=node.type)
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)

    def visitPackArraySlice(self, node, slice):
        ndata=slice.data.pack(node.type, len(node.type.dims))
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)

    def visitInsertDimSlice(self,node,slice):
        ndata = slice.data.insertDim(node.matchpoint,node.newdim)
        return slice.modify(data=ndata,dims=node.dims)

    def visitPackListSlice(self, node, slice):
        ndata=slice.data.pack(node.type, len(node.type.dims))
        ndata=ndata.map(list,res_type=node.type)
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)

    def visitUnpackTupleSlice(self,node,slice):
        func = operator.itemgetter(node.tuple_idx)
        ndata = slice.data.map(func,res_type=node.type)
        return slice.modify(data=ndata,rtype=node.type,name=node.name)
 
    def visitPackTupleSlice(self,node, slices):
        ndata = nested_array.co_mapseq(speedtuplify,[slice.data for slice in slices],res_type=node.type)
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)
    
    def visitHArraySlice(self,node, slices):
        ndata = nested_array.co_mapseq(speedarrayify,[slice.data for slice in slices],dtype=node.type.subtypes[0].toNumpy(), res_type=node.type)
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)

    def visitEnsureCommonDimSlice(self,node,slice,compare_slice):
        checkdim = node.dims[node.checkpos]
        selfshape = slice.data.getDimShape(node.checkpos)

        otherpos = compare_slice.dims.index(checkdim)
        othershape = compare_slice.data.getDimShape(otherpos)
        assert numpy.all(selfshape == othershape), "Dimension mismatch in " + str(checkdim) + ":" + str(selfshape) + " != " + str(othershape)

        ndata = slice.data.replaceDim(node.checkpos,checkdim)
        return slice.modify(data=ndata,dims=node.dims)

    def visitBroadcastSlice(self,node,slice,compare_slices):
        repeat_dict = {}
        dim_dict = {}
        bcpos = 0
        for pos,planelem in enumerate(node.plan):
            if(planelem == BCEXIST):
                dimpos = compare_slices[bcpos].dims.index(node.bcdims[pos])
                repeat_dict[pos] = compare_slices[bcpos].data.getDimShape(dimpos)
                dim_dict[pos] = compare_slices[bcpos].dims[dimpos]
                bcpos += 1
            elif(planelem == BCCOPY):
                pass
            else:
                raise RuntimeError, "Unknown broadcast plan element: " + str(planelem)
        ndata = slice.data.broadcast(repeat_dict,dim_dict)
        return slice.modify(data=ndata,dims=node.dims)

    def visitBinElemOpSlice(self,node, slices):
        func = node.oper.exec_funcs["py"]
        ndata = nested_array.co_mapseq(func,[slice.data for slice in slices],
                                       type1=slices[0].type,type2=slices[1].type,
                                       typeo=node.type,res_type=node.type,op=node.op,
                                       bc_allow=True)
        
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)

    def visitFixate(self,node,slices):
        res = []
        for cur_slice, slice in zip(node._slices, slices):
            ndata = slice.data.getStructuredData()
            nslice = slice.modify(data=ndata)
            nslice.setSource(cur_slice)
            res.append(nslice)
        return res
    
    def castto_any(self,castname,node,slice):
        return slice.data.mapseq(lambda x:x,res_type=node.type)

    def castnumbers_numbers(self,castname,node,slice):
        return slice.data.mapseq(lambda x:x,res_type=node.type)

#util funcs
def speedtuplify(seqs):
    nseq = cutils.darray(zip(*seqs))
    return nseq

def speedarrayify(seqs,dtype):
    nseq = numpy.array(seqs,dtype).T
    return nseq

