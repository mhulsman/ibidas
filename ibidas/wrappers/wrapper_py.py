import operator
from itertools import chain
from collections import defaultdict
import numpy
import sys

import wrapper
from ..constants import *
from ..utils.multi_visitor import VisitorFactory, DirectVisitorFactory, NF_ERROR, NF_ELSE
from ..itypes import rtypes, dimpaths
from ..passes import manager, create_graph, serialize_exec
from .. import ops

from ..utils.multi_visitor import VisitorFactory, DirectVisitorFactory, NF_ELSE

_delay_import_(globals(),"..representor")
_delay_import_(globals(),"..utils","util","nestutils","cutils","nested_array","context")
_delay_import_(globals(),"..itypes","detector","type_attribute_freeze","convertors","dimensions")
_delay_import_(globals(),"..repops_slice")
_delay_import_(globals(),"..repops_dim")
_delay_import_(globals(),"..utils.missing","Missing")


class PyRepresentor(wrapper.SourceRepresentor):
    pass

def rep(data=None, dtype=None, unpack=True, name=None):
    """Packs python data structures into a :py:class:`ibidas.representor.Representor` object.
        
       :param data: any python object
       :param dtype: type of ``data``. See :py:func:`createType` for allowed string formats
                     (default=autodetect).
       :type dtype: :py:class:`str` or :py:class:`ibidas.itypes.rtypes.Type`
       :param unpack: determines if data object is unpacked. 
                      For example, unpacking an array means that subsequent operations are performed 
                      on the elements instead of on the array structure
       :param name: Name of the main slice, representing the data (default='data').
       :type name: :py:class:`str`, lower case
      
       Examples:
            Let the python data that needs to be represented be a simple table (collection of tuples):
            
            >>> data = [('gene1',0.5),('gene2',0.3),('gene100',0.9)]

            The, to load using type autodetection, use:

            >>> r = rep(data)
            Slices: f0       f1     
            Types:  bytes[7] real64 
            Dims:   d1:3     .   
            
            Specifying type directly:

            >>> r = rep(data,"[genes]<(name:string, value:real64)")
            Slices: name    value
            Types:  string  real64
            Dims:   genes:* .

            Effect of setting unpack to False:

            >>> r = rep(data,"[genes]<(name:string, value:real64)", unpack=False)
            Slices: data                                  
            Types:  [genes:*]:(name=string, value=real64) 
            Dims: 
            
            Specifying root slice name:

            >>> r = rep(data,"[genes]<(name:string, value:real64)", unpack=False, name="gene_table")
            Slices: gene_table                                  
            Types:  [genes:*]:(name=string, value=real64) 
            Dims: 

    """
    if(not dtype is None):
        if(isinstance(dtype,str)):
            dtype = rtypes.createType(dtype)
        dtype = dtype._setNeedConversionRecursive(True)
    else:
        det = detector.Detector()
        det.process(data)
        dtype = det.getType()
        if(dtype == rtypes.unknown):
            dtype = rtypes.TypeAny(True)

    if(name is None):
        name = "data"

    data_slice = ops.DataOp(data,name=name,rtype=dtype)
    data_slice = ops.ensure_converted(data_slice)
    
    while(unpack and data_slice.type.__class__ is rtypes.TypeArray):
        data_slice = ops.ensure_converted(ops.UnpackArrayOp(data_slice))

    res = wrapper.SourceRepresentor()
    res._initialize((data_slice,)) 

    if(unpack and isinstance(res.getType(), rtypes.TypeTuple)):
        res = repops_slice.UnpackTuple(res)
    
    return res


class ResultOp(ops.DataOp):
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

    def modify(self,data=NOVAL, name=NOVAL, rtype=NOVAL, dims=NOVAL, bookmarks=NOVAL):
        if(name is NOVAL):
            name = self.name
        if(rtype is NOVAL):
            rtype = self.type
        if(dims is NOVAL):
            dims = self.dims
        if(data is NOVAL):
            data = self.data
        if(bookmarks is NOVAL):
            bookmarks = self.bookmarks

        return ResultOp(data, name, rtype, dims, bookmarks)

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
                
            try:
                res = self.visit(command, **param_kwds)
            except Exception, e:
                exc_info = sys.exc_info()
                if(debug_mode):
                    try:
                        from ..passes.cytoscape_vis import DebugVisualizer
                        DebugVisualizer.run(query,run_manager)
                    except Exception:
                        pass
                raise exc_info[1], None, exc_info[2]
            if(debug_mode):
                self.graph.na["output"][command] = str(res)

            arguments[command_id] = res
            use_counters[command_id] = param_usecount[command]
            arguments[use_counters == 0] = None
        return res

    def visitelse(self,node, **kwargs):
        raise RuntimeError, "Unknown node type encountered: " + node.__class__.__name__ + " with  " + str(kwargs)

    def visitExtendOp(self,node, *args, **kwargs):
        return node.py_exec(*args, **kwargs)

    def visitDataOp(self,node):
        if(isinstance(node.data, nested_array.NestedArray)):
            ndata = node.data.copy()
        else:
            ndata = nested_array.NestedArray(node.data,node.type)
        return ResultOp.from_slice(ndata,node)

    def visitConvertOp(self,node,slice):
        ndata = slice.data.mapseq(node.convertor.convert,
                                node.type,res_type=node.type)
        return slice.modify(data=ndata)
   
    def visitCastOp(self,node,slice):
        ndata = self.cast(node.cast_name,node,slice)
        return slice.modify(data=ndata,rtype=node.type)

    def visitChangeNameOp(self,node,slice):
        return slice.modify(name = node.name)

    def visitChangeDimPathOp(self,node,slice):
        return slice.modify(dims=node.dims)

    def visitChangeBookmarkOp(self,node,slice):
        return slice.modify(bookmarks=node.bookmarks)

    def visitDetectTypeOp(self,node,slice):
        if(slice.type == rtypes.unknown):
            det = detector.Detector()
            det.setParentDimensions(node.dims)
            det.processSeq(slice.data.flat())
            return slice.modify(rtype=det.getType())
        else:
            return slice
    
    def visitUnpackArrayOp(self,node,slice):
        ndata=slice.data.unpack(node.unpack_dims, subtype=node.type)
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)

    def visitPackArrayOp(self, node, slice):
        ndata=slice.data.pack(node.type, len(node.type.dims))
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)

    def visitInsertDimOp(self,node,slice):
        ndata = slice.data.insertDim(node.matchpoint)
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)

    def visitPermuteDimsOp(self, node, slice):
        ndata=slice.data.permuteDims(node.permute_idxs)
        return slice.modify(data=ndata,dims=node.dims,rtype=node.type)

    def visitSplitDimOp(self, node, slice):
        ndata=slice.data.splitDim(node.pos,node.lshape,node.rshape)
        return slice.modify(data=ndata,dims=node.dims,rtype=node.type)

    def visitPackListOp(self, node, slice):
        ndata=slice.data.pack(node.type, len(node.type.dims))
        func = lambda x: x.tolist()
        ndata=ndata.map(func,res_type=node.type)
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)

    def visitToPythonOp(self,  node, slice):
        func = lambda x: convertors.rpc_convertor.execConvert(slice.type,x)
        ndata = slice.data.mapseq(func,res_type=node.type)
        return slice.modify(data=ndata, rtype=node.type)

    def visitUnpackTupleOp(self,node,slice):
        func = operator.itemgetter(node.tuple_idx)
        ndata = slice.data.map(func,res_type=node.type)
        return slice.modify(data=ndata,rtype=node.type,name=node.name)
 
    def visitPackTupleOp(self,node, slices):
        ndata = nested_array.co_mapseq(speedtuplify,[slice.data for slice in slices],res_type=node.type)
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)
    
    def visitHArrayOp(self,node, slices):
        ndata = nested_array.co_mapseq(speedarrayify,[slice.data for slice in slices],dtype=node.type.subtypes[0].toNumpy(), res_type=node.type)
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)

    def visitEnsureCommonDimOp(self,node,slice,compare_slice):
        checkdim = node.dims[node.checkpos]
        selfshape = slice.data.getDimShape(node.checkpos)

        otherpos = compare_slice.dims.index(checkdim)
        othershape = compare_slice.data.getDimShape(otherpos)
        assert numpy.all(selfshape == othershape), "Dimension mismatch in " + str(checkdim) + ":" + str(selfshape) + " != " + str(othershape)

        return slice.modify(rtype=node.type,dims=node.dims)

    def visitShapeOp(self, node, slice):
        d = slice.data.getDimShape(node.pos)
        ndata = nested_array.NestedArray(d,node.type)
        return slice.modify(ndata,rtype=node.type,dims=node.dims,name=node.name)

    def visitFreezeOp(self, node, slice):
        func = lambda x: type_attribute_freeze.freeze_protocol.execFreeze(slice.type,x)
        ndata = slice.data.mapseq(func,res_type=node.type)
        return slice.modify(data=ndata)

    def visitBroadcastOp(self,node,slice,compare_slices):
        repeat_dict = {}
        bcpos = 0
        for pos,planelem in enumerate(node.plan):
            if(planelem == BCEXIST):
                dimpos = compare_slices[bcpos].dims.index(node.bcdims[pos])
                
                rshape = compare_slices[bcpos].data.getDimShape(dimpos)
                #if it is a variable shape, keep only shape dims that are variable
                if(not isinstance(rshape,int)):
                    dim = compare_slices[bcpos].dims[dimpos]
                    rshape = remove_independent(rshape,dim)
                    rshape = add_independent(rshape,node.dims[pos])
                repeat_dict[pos] = rshape
                bcpos += 1
            elif(planelem == BCCOPY):
                pass
            else:
                raise RuntimeError, "Unknown broadcast plan element: " + str(planelem)
        ndata = slice.data.broadcast(repeat_dict)
        return slice.modify(data=ndata,dims=node.dims)


    def visitFilterOp(self,node, slice, constraint):
        ndata = nested_array.co_map(speedfilter,[slice.data, constraint.data],
                                       has_missing = node.has_missing,ctype=constraint.type,
                                       res_type=node.type, bc_allow=True)
        return slice.modify(data=ndata,rtype=node.type)

    def visitFlatAllOp(self, node, slice):
        ndata = slice.data.mergeAllDims()
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)
    
    def visitFlatDimOp(self, node, slice):
        ndata = slice.data.mergeDim(node.flatpos-1)
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)


    def visitGroupIndexOp(self, node, slices):
        ndata = nested_array.co_map(groupindex,[slice.data for slice in slices],
                                        res_type = node.type, bc_allow=False)
        
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)

    def visitUnaryFuncElemOp(self,node, slice):
        try:
            func = getattr(self, node.sig.name + node.funcname)
        except AttributeError:
            func = getattr(self, node.sig.name + "General")
        

        ndata = slice.data.mapseq(func,type_in=slice.type,type_out=node.type,
                                  res_type=node.type,op=node.funcname, **node.kwargs)
        return slice.modify(data=ndata,rtype=node.type)

    def visitUnaryFuncSeqOp(self,node, slice):
        try:
            func = getattr(self, node.sig.name + node.funcname)
        except AttributeError:
            func = getattr(self, node.sig.name + "General")
       
        ndata = slice.data
        if(node.packdepth > 2):
            ndata,nshapes = ndata.mergeLastDims(node.packdepth-2)
            dim = dimensions.Dim(UNDEFINED,(True,))
            ldims = dimpaths.DimPath(slice.dims[-node.packdepth],dim)
        else:
            ldims = slice.dims[-node.packdepth:]
        
        ndata = ndata.pack(slice.type, min(node.packdepth,2))
        ndata = ndata.mapseq(func,type_in=slice.type,type_out=node.type,
                                  res_type=node.type,op=node.funcname, packdepth=node.packdepth, **node.kwargs)
        ndata = ndata.unpack(ldims, subtype=node.type)
        if(node.packdepth > 2):
            ndata = ndata.splitLastDim(nshapes)

        return slice.modify(data=ndata,rtype=node.type)

       
    def visitUnaryFuncAggregateOp(self,node, slice):
        try:
            func = getattr(self, node.sig.name + node.funcname)
        except AttributeError:
            func = getattr(self, node.sig.name + "General")
        ndata = slice.data
        if(node.packdepth > 1):
            if(node.packdepth > 2):
                ndata,nshapes = ndata.mergeLastDims(node.packdepth-2)
            ndata = ndata.pack(slice.type, 2)
        else:
            ndata = ndata.pack(slice.type)
        ndata = ndata.mapseq(func,type_in=slice.type,type_out=node.type,
                                  res_type=node.type,op=node.funcname,packdepth=node.packdepth, **node.kwargs)
        
        if(node.packdepth > 1):
            dim = dimensions.Dim(UNDEFINED,(True,))
            ndata = ndata.unpack(dimpaths.DimPath(dim), subtype=node.type)
            if(node.packdepth > 2):
                nshapes = nested_array.drop_prev_shapes_dim(ndata,nshapes)
                ndata = ndata.splitLastDim(nshapes)
            
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)



    def visitBinFuncElemOp(self,node, slices):
        try:
            func = getattr(self, node.sig.name + node.funcname)
        except AttributeError:
            func = getattr(self, node.sig.name + "General")

        ndata = nested_array.co_mapseq(func,[slice.data for slice in slices],
                                       type1=slices[0].type,type2=slices[1].type,
                                       typeo=node.type,res_type=node.type,op=node.funcname,
                                       bc_allow=node.allow_partial_bc)
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)

    def visitEquiJoinIndexOp(self, node, slices):
        ndatas = nested_array.co_map(joinindex,[slice.data for slice in slices],
                                       jointype=node.jointype,
                                       res_type=(node.results[0].type,node.results[1].type),
                                       bc_allow=False)
        return slices[0].modify(data=ndatas,name=None,rtype=None,dims=None, bookmarks=None)

    def visitSelectOp(self, node, slice):
        return slice.modify(data=slice.data[node.index],name=node.name, rtype=node.type, dims=node.dims, bookmarks=node.bookmarks)

    def visitFixate(self,node,slices):
        res = []
        for cur_slice, slice in zip(node._slices, slices):
            ndata = slice.data.getStructuredData()
            nslice = slice.modify(data=ndata)
            nslice.setSource(cur_slice)
            res.append(nslice)
        return res
    
    def visitGather(self,node,slices):
        res = []
        for cur_slice, slice in zip(node._slices, slices):
            slice.setSource(cur_slice)
            res.append(slice)
        return res
   
    def castto_any(self,castname,node,slice):
        return slice.data.mapseq(lambda x:x,res_type=node.type)

    def castnumbers_numbers(self,castname,node,slice):
        return slice.data.mapseq(lambda x:x,res_type=node.type)


    def withinWithin(self, data, type1, type2, typeo, op):
        data1,data2 = data
        res = [elem in arr for elem,arr in zip(data1,data2)]
        return cutils.darray(res,bool)


    def simple_arithGeneral(self, data, type1, type2, typeo, op):
        data1,data2 = data
        if(data1 is Missing or data2 is Missing):
            return Missing
        return numpy_arith[op](data1, data2, sig=typeo.toNumpy())
    
    def string_add_stringAdd(self, data, type1, type2, typeo, op):
        data1,data2 = data
        if(data1 is Missing or data2 is Missing):
            return Missing
        return numpy_arith[op](numpy.cast[object](data1), numpy.cast[object](data2), sig=typeo.toNumpy())
    
    def array_add_arrayAdd(self, data, type1, type2, typeo, op):
        data1,data2 = data
        if(data1 is Missing or data2 is Missing):
            return Missing
        res = []
        dtype = typeo.toNumpy()
        for lelem, relem in zip(data1,data2):
            res.append(numpy.concatenate([numpy.cast[dtype](lelem),numpy.cast[dtype](relem)],axis=0))
        return cutils.darray(res,object)

    def simple_cmpGeneral(self, data, type1, type2, typeo, op):
        #a numpy bug gives all true arrays when using
        #bool as outtype in comparison
        return numpy_cmp[op](data[0], data[1])

    def string_cmpGeneral(self, data, type1, type2, typeo, op):
        #a numpy bug gives NotImplemented when performing operations,
        #such as "numpy.equal" on string arrays
        #so use direct operations ("__eq__")
        op = python_op[op]
        res = getattr(data[0], op)(data[1])
        if(res is NotImplemented):
            res = getattr(data[1], reverse_op[op])(data[0])
        assert not res is NotImplemented, "Not implemented error in stringstringGeneral for " \
                                            + str(op) + " and " + str(type1) + ", " + str(type2)
        return res

    def numberGeneral(self, data, type_in, type_out, op):
        util.debug_here()
        return numpy_unary_arith[op](data, sig=type_out.toNumpy())

    def corrCorr(self, data, type_in, type_out, op):
        intype = type_in.toNumpy()
        res = []
        for elem in data:
            if(len(elem.shape) < 2):
                elem = cutils.darray(list(elem),intype,2,2)
            res.append(numpy.corrcoef(elem))
        return cutils.darray(res,object,1,1)

    def eachEach(self, data, type_in, type_out, op, eachfunc):
        if(isinstance(eachfunc,context.Context)):
            return cutils.darray([context._apply(eachfunc,elem) for elem in data],type_out.toNumpy())
        else:
            return cutils.darray([eachfunc(elem) for elem in data],type_out.toNumpy())

    def sortableArgSort(self, data, type_in, type_out, op, packdepth, descend=False):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        if(len(data.shape) < 2):
            if(descend):
                res = cutils.darray([numpy.flipud(numpy.argsort(row,axis=0)) for row in data],object)
            else:
                res = cutils.darray([numpy.argsort(row,axis=0) for row in data],object)
        else:
            res = numpy.argsort(data,axis=1)
            if(descend):
                res = res[:,::-1,...]
        return res
   
    def any_nodepPos(self, data, type_in, type_out, op, packdepth):
        dtype = type_out.toNumpy()
        if(len(data.shape) == 1):
            if(packdepth == 1):
                xres = [numpy.arange(len(row),dtype=dtype) for row in data]
            else:
                xres = []
                for row in data:
                    res = numpy.arange(len(row),dtype=dtype)
                    if(len(row.shape) > 1):
                        res = numpy.repeat(res,row.shape[1]).reshape(row.shape)
                    else:
                        subres = []
                        for r,d in zip(res.ravel(),row.ravel()):
                            subres.append(cutils.darray([r] * len(d),dtype))
                        res = cutils.darray(subres,object)
                    xres.append(res)
            xres = cutils.darray(xres)
        elif(len(data.shape) == 2):
            xres = numpy.tile(numpy.arange(data.shape[1],dtype=dtype),data.shape[0]).reshape(data.shape[:2])
            if(packdepth > 1):
                res = []
                for r,d in zip(xres.ravel(),data.ravel()):
                    res.append(cutils.darray([r] * len(d),dtype))
                xres = cutils.darray(res,object,1,1).reshape(data.shape)
        else:
            assert len(data.shape) == 3, "Unexpected data shape"
            r =  numpy.tile(numpy.arange(data.shape[1],dtype=dtype),data.shape[0]).reshape(data.shape[:2])
            xres = numpy.repeat(r,data.shape[2]).reshape(data.shape)
           
        return xres

    def countCount(self, data, type_in, type_out, op, packdepth):
        dtype = type_out.toNumpy()
        if(len(data.shape) == 1):
            return cutils.darray([len(row) for row in data],dtype)
        else:
            return cutils.darray([data.shape[1]] * data.shape[0],dtype)
    
    def setGeneral(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        dtype = type_out.toNumpy()
        if(packdepth > 1):
           return cutils.darray([[set(subrow) for subrow in row.transpose()] for row in data],dtype,2,2)
        else:
           return cutils.darray([set(row) for row in data],dtype)
    
    def arrayarraySum(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        if(packdepth > 1):
            return cutils.darray([[numpy.concatenate(list(subrow),axis=0) for subrow in row.transpose()] for row in data],object,2,2)
        else:
            return cutils.darray([numpy.concatenate(list(row),axis=0) for row in data],object)
 
    def fixdimGeneral(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        func = numpy_dimfuncs[op]
        dtype = type_out.toNumpy()
        if(len(data.shape) < 2):
            if(packdepth > 1):
                dtype = object
            return cutils.darray([func(row,axis=0) for row in data],dtype)
        else:
            return func(data,axis=1)
      
numpy_cmp = {'Equal':numpy.equal,#{{{
            'NotEqual':numpy.not_equal,
            'LessEqual':numpy.less_equal,
            'GreaterEqual':numpy.greater_equal,
            'Less':numpy.less,
            'Greater':numpy.greater}

numpy_arith = { 'Add':numpy.add,
                'Subtract':numpy.subtract,
                'Multiply':numpy.multiply,
                'Modulo':numpy.mod,
                'Divide':numpy.divide,
                'FloorDivide':numpy.floor_divide,
                'And':numpy.bitwise_and,
                'Or':numpy.bitwise_or,
                'Xor':numpy.bitwise_xor,
                'Power':numpy.power
                }
numpy_dimfuncs = {
    'Max':numpy.max,
    'Min':numpy.min,
    'Mean':numpy.mean,
    'Any':numpy.any,
    'All':numpy.all,
    'Sum':numpy.sum,
    'Median':numpy.median,
    'ArgMin':numpy.argmin,
    'ArgMax':numpy.argmax
    }

python_op = {'Equal':'__eq__',
             'NotEqual':'__ne__',
             'LessEqual':'__le__',
             'GreaterEqual':'__ge__',
             'Less':'__lt__',
             'Greater':'__gt__',
             'Add':'__add__',
             'Subtract':'__sub__',
             'Multiply':'__mul__',
             'Modulo':'__mod__',
             'Divide':'__div__',
             'FloorDivide':'__floordiv__',
             'And':'__and__',
             'Or':'__or__',
             'Xor':'__xor__',
             'Power':'__pow__'}

numpy_unary_arith = {
    "Invert":numpy.invert,
    "Negative":numpy.negative,
    "Abs":numpy.abs
    }

reverse_op = {'__eq__':'__eq__',
            '__ne__':'__ne__',
            '__le__':'__ge__',
            '__ge__':'__le__',
            '__lt__':'__gt__',
            '__gt__':'__lt__',
            '__add__':'__radd__',
            '__radd__':'__add__',
            '__sub__':'__rsub__',
            '__rsub__':'__sub__',
            '__mul__':'__rmul__',
            '__rmul__':'__mul__',
            '__mod__':'__rmod__',
            '__rmod__':'__mod__',
            '__div__':'__rdiv__',
            '__rdiv__':'__div__',
            '__floordiv__':'__rfloordiv__',
            '__rfloordiv__':'__floordiv__',
            '__and__':'__rand__',
            '__rand__':'__and__',
            '__or__':'__ror__',
            '__ror__':'__or__',
            '__xor__':'__rxor__',
            '__rxor__':'__xor__',
            '__pow__':'__rpow__',
            '__rpow__':'__pow__'
            }#}}}

#util funcs
def speedtuplify(seqs):
    nseq = cutils.darray(zip(*seqs))
    return nseq

def speedarrayify(seqs,dtype):
    nseq = numpy.array(seqs,dtype).T
    return nseq


def speedfilter(seqs,has_missing, ctype):
    data,constraint = seqs
    if(has_missing):
        if(isinstance(ctype,rtypes.TypeArray)):
            if(isinstance(ctype.subtypes[0],rtypes.TypeBool)):
                data = data.ravel()
                res = []
                for pos, elem in enumerate(constraint.ravel()):
                    if(elem is Missing):
                        res.append(Missing)
                    elif(elem is True):
                        res.append(data[pos])
                res = cutils.darray(res,object)
            else:#indices
                data = data.ravel()
                res = []
                for elem in constraint.ravel():
                    if(elem is Missing):
                        res.append(Missing)
                    else:
                        res.append(data[elem])
                res = cutils.darray(res,object)
        else:
            if(constraint is Missing):
                return Missing
            else:
                return data[constraint]
    else:
        res = data[constraint]
    return res

def ensure_fixeddims(seqs,packdepth,dtype):
    if(packdepth > 1):
        if(len(seqs.shape) == 1):
            res = []
            for seq in seqs:
                if(len(seq.shape) >= 2):
                    res.append(seq)
                else:
                    res.append(cutils.darray(seq,dtype,100000,2))
            res = cutils.darray(res,object)
        elif(len(seqs.shape) == 2):
            res = cutils.darray(seqs.tolist(),dtype,1000,3)
        else:
            res = seqs
    else:
        res = seqs
    return res


def groupindex(data):
    indexes = []
    if(len(data) > 1):
        for dcol in data:
            index = {}
            pos = 0 
            for elem in dcol:
                if not elem in index:
                    index[elem] = pos
                    pos += 1
            indexes.append(index)
        
        data_dict = defaultdict(list)
        for pos, elems in enumerate(zip(*data)):
            data_dict[elems].append(pos)
       
        shape = [len(index) for index in indexes]
        indexdata = [cutils.darray([],int)] * numpy.prod(shape)
        indexdata = cutils.darray(indexdata, object, 1).reshape(tuple(shape))
        
        for key, posses in data_dict.iteritems():
            loc = tuple([index[keypart] for index, keypart in zip(indexes,key)])
            indexdata[loc] = cutils.darray(posses, int)
    else:
        data_dict = defaultdict(list)
        for pos, elems in enumerate(*data):
            if(elems is Missing):
                data_dict["__TEMP__" + str(elems.__hash__())].append(pos)
            else:
                data_dict[elems].append(pos)
        indexdata = [cutils.darray(elem, int) for elem in data_dict.values()]
        indexdata = cutils.darray(indexdata, object, 1)
   
    return indexdata


def create_elem_pos_dict(data):
    res = {}
    for pos, elem in enumerate(data):
        if elem in res:
            res[elem].append(pos)
        else:
            res[elem] = [pos]
    return res

def joinindex(data, jointype):
    lelempos = create_elem_pos_dict(data[0])
    relempos = create_elem_pos_dict(data[1])

    tlpos = []
    trpos = []
    leftouter = (jointype == "left" or jointype == "full")
    rightouter = (jointype == "right" or jointype == "full")
    for elem,lpos in lelempos.iteritems():
        if not elem in relempos:
            if(leftouter):
                tlpos.extend(lpos)
                trpos.extend([Missing] * len(lpos))
            continue
        rpos = relempos[elem]
        if(len(lpos) == 1):
            if len(rpos) == 1:
                tlpos.append(lpos[0])
                trpos.append(rpos[0])
            else:
                tlpos.extend(lpos * len(rpos))
                trpos.extend(rpos)
        else:
            if len(rpos) == 1:
                tlpos.extend(lpos)
                trpos.extend(rpos * len(lpos))
            else:
                tlpos.extend(lpos * len(rpos))
                trpos.extend(numpy.repeat(rpos,len(lpos)))
    
    if(rightouter):
        unused_keys = list(set(range(len(data[1]))) - set(trpos))
        tlpos.extend([Missing] * len(unused_keys))
        trpos.extend(unused_keys)
        ldtype = object
    else:
        ldtype = int

    if(leftouter):
        rdtype = object
    else:
        rdtype = int

    return (cutils.darray(tlpos,ldtype), cutils.darray(trpos,rdtype))

def remove_independent(data,dim):
    wx = [0] * len(data.shape)
    for pos, dep in enumerate(dim.dependent):
        #FIXME: handle variable prev dims (move to nested_array)
        if dep and len(wx) > pos:
            wx[-(pos+1)] = slice(None,None)
    data = data[wx]
    return data

def add_independent(data,dim):
    wx = list(data.shape)[::-1]
    for pos, dep in enumerate(dim.dependent):
        if not dep:
            wx.insert(pos,1)
    data = numpy.reshape(data,wx[::-1])
    return data

   
