import operator
from itertools import chain
from collections import defaultdict
import numpy
import sys
import cPickle
import time

import wrapper
from ..constants import *
from ..utils.multi_visitor import VisitorFactory, DirectVisitorFactory, NF_ERROR, NF_ELSE
from ..itypes import rtypes, dimpaths
from ..passes import manager, create_graph, serialize_exec
from .. import ops

from ..utils.multi_visitor import VisitorFactory, DirectVisitorFactory, NF_ELSE

_delay_import_(globals(),"..representor")
_delay_import_(globals(),"..utils","util","cutils","nested_array","context")
_delay_import_(globals(),"..itypes","detector","type_attribute_freeze","convertors","dimensions")
_delay_import_(globals(),"..repops_slice")
_delay_import_(globals(),"..repops_dim")
_delay_import_(globals(),"..utils.missing","Missing")


class PyRepresentor(wrapper.SourceRepresentor):
    def __init__(self, slices):
        self._initialize(slices)

def Rep(data=None, dtype=None, unpack=True, name=None):
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

            >>> r = Rep(data)
            Slices: f0       f1     
            Types:  bytes[7] real64 
            Dims:   d1:3     .   
            
            Specifying type directly:

            >>> r = Rep(data,"[genes]<(name:string, value:real64)")
            Slices: name    value
            Types:  string  real64
            Dims:   genes:* .

            Effect of setting unpack to False:

            >>> r = Rep(data,"[genes]<(name:string, value:real64)", unpack=False)
            Slices: data                                  
            Types:  [genes:*]:(name=string, value=real64) 
            Dims: 
            
            Specifying root slice name:

            >>> r = Rep(data,"[genes]<(name:string, value:real64)", unpack=False, name="gene_table")
            Slices: gene_table                                  
            Types:  [genes:*]:(name=string, value=real64) 
            Dims: 

    """
    if(not dtype is None):
        if(isinstance(dtype,basestring)):
            dtype = rtypes.createType(dtype)
    else:
        det = detector.Detector()
        det.process(data)
        dtype = det.getType()
        if(dtype == rtypes.unknown):
            dtype = rtypes.TypeAny(True)

    if(name is None):
        res = util.find_names([data])
        if res:
            name = res[0]
        else:            
            name = "data"

    data_slice = ops.DataOp(data,name=name,rtype=dtype)
    
    while(unpack and data_slice.type.__class__ is rtypes.TypeArray):
        data_slice = ops.UnpackArrayOp(data_slice)

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
        self.bookmarks = source.bookmarks
        self.name = source.name
        self.dims = source.dims

    def __repr__(self):
        res = "\nName: " + str(self.name) + ", Type:   " + str(self.type) + ", Dims: " + str(self.dims) 
        try:
            r = str(self.data)
            r.decode('utf-8')
        except UnicodeDecodeError:
            return res
        res += '\n' + r
        return res

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
                
            if(debug_mode):
                start = time.time()
            try:
                res = self.visit(command, **param_kwds)
            except Exception, e:
                exc_info = sys.exc_info()
                if(debug_mode):
                    try:
                        from ..passes.cytoscape_vis import DebugVisualizer
                        DebugVisualizer.run(query,run_manager)
                    except Exception:
                        raise
                raise exc_info[1], None, exc_info[2]
            if(debug_mode):
                rtime = time.time() - start
                self.graph.na["output"][command] = str(res)
                self.graph.na["time"][command] = rtime
                self.graph.na["time_readable"][command] = util.format_runtime(rtime)

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

    def visitCastOp(self,node,slice):
        ndata = self.cast(node.cast_name,node,slice)
        return slice.modify(data=ndata,rtype=node.type)

    def visitChangeNameOp(self,node,slice):
        return slice.modify(name = node.name)

    def visitChangeDimPathOp(self,node,slice):
        return slice.modify(dims=node.dims)

    def visitChangeBookmarkOp(self,node,slice):
        return slice.modify(bookmarks=node.bookmarks)
    
    def visitChangeDimOp(self,node,slice):
        return slice.modify(rtype=node.type,dims=node.dims)

    def visitDetectFixedShapesOp(self,node,slice):
        data = slice.data.flat()
        if len(data) == 0:
            raise RuntimeError, "Cannot determine dim shape from empty data. Please cast dim shape of fixed dims."
        ntype = detect_shape(data[0],slice.type)
        return slice.modify(rtype=ntype)
    
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
        ndata=slice.data.pack(node.type, len(node.pack_dims))
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
        ndata=slice.data.pack(node.type, len(node.pack_dims))
        func = lambda x: x.tolist()
        ndata=ndata.map(func,res_type=node.type)
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)

    def visitToPythonOp(self,  node, slice):
        func = lambda x: convertors.rpc_convertor.execConvert(slice.type,x)
        ndata = slice.data.mapseq(func,res_type=node.type)
        return slice.modify(data=ndata, rtype=node.type)

    def visitUnpackTupleOp(self,node,slice):
        if(isinstance(slice.type,rtypes.TypeRecordDict)):
            didx = slice.type.fieldnames[node.tuple_idx]
            if(node.type.has_missing):
                func = lambda x: x.get(didx,Missing)
            else:
                func = operator.itemgetter(didx)
        elif(isinstance(slice.type,rtypes.TypeIndexDict)):
            if node.tuple_idx == 0:
                func = lambda x: x.keys()
            else:
                func = lambda x: x.values()
        else:
            if(node.type.has_missing):
                def func(x):
                    if node.tuple_idx >= len(x):
                        return Missing
                    else:
                        return x[node.tuple_idx]
            else:
                func = operator.itemgetter(node.tuple_idx)
        ndata = slice.data.map(func,res_type=node.type)
        return slice.modify(data=ndata,rtype=node.type,name=node.name)
 
    def visitPackTupleOp(self,node, slices):
        ndata = nested_array.co_mapseq(speedtuplify,[slice.data for slice in slices],res_type=node.type)
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)
   
    def visitPackIndexDictOp(self, node, slices):
        names = node.type.fieldnames
        def speed_index_dictify(x):
            return dict(zip(*x))
        ndata = nested_array.co_map(speed_index_dictify,[slice.data for slice in slices],res_type=node.type)
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)
       
    def visitTakeOp(self, node, slices):
        ndata = nested_array.co_mapseq(speeddictindex,[slice.data for slice in slices],res_type=node.type, 
                                        dtype=node.type.toNumpy(), allow_missing=node.allow_missing, keep_missing = node.keep_missing)
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)

    def visitPackDictOp(self,node, slices):
        names = node.type.fieldnames
        if node.with_missing or not any([slice.type.has_missing for slice in slices]):
            def speeddictify(x):
                return util.darray([dict(zip(names,row)) for row in zip(*x)])
        else:
            def speeddictify(x):
                non_hasmissing = []
                for pos, slice in enumerate(slices):
                    if not slice.type.has_missing:
                        non_hasmissing.append(pos)
                if(non_hasmissing):
                    nx = util.darray(x)[non_hasmissing] 
                    snames = util.darray(names)[non_hasmissing]
                    d = [dict(zip(snames,row)) for row in zip(*nx)]
                else:
                    d = [{} for i in xrange(len(x[0]))]
                for pos, slice in enumerate(slices):
                    if(pos in non_hasmissing):
                        continue
                    name = names[pos]
                    val = x[pos]
                    for rowpos in xrange(len(val)):
                        elem = val[rowpos]
                        if not elem is Missing:
                            d[rowpos][name] =elem
                return util.darray(d)

        ndata = nested_array.co_mapseq(speeddictify,[slice.data for slice in slices],res_type=node.type)
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
            elif(planelem == BCCOPY or planelem == BCSOURCE):
                pass
            else:
                raise RuntimeError, "Unknown broadcast plan element: " + str(planelem)
        ndata = slice.data.broadcast(repeat_dict)
        return slice.modify(data=ndata,dims=node.dims)


    def visitFilterOp(self,node, slice, constraint):
        ndata = nested_array.co_map(speedfilter,[slice.data, constraint.data],
                                       has_missing = node.has_missing,ctype=constraint.type,stype=node.type,
                                       res_type=node.type)
        return slice.modify(data=ndata,rtype=node.type, dims=node.dims)

    def visitFlatAllOp(self, node, slice):
        ndata = slice.data.mergeAllDims()
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)
    
    def visitFlatDimOp(self, node, slice):
        ndata = slice.data.mergeDim(node.flatpos-1,result_fixed=len(node.dims[node.flatpos-1].dependent) == 0)
        return slice.modify(data=ndata,rtype=node.type,dims=node.dims)


    def visitGroupIndexOp(self, node, slices):
        ndata = nested_array.co_map(groupindex,[slice.data for slice in slices],
                                        res_type = node.type)
        
        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)

    def visitUnaryFuncElemOp(self,node, slice):
        try:
            func = getattr(self, node.sig.name + node.funcname)
        except AttributeError:
            func = getattr(self, node.sig.name + "General")
        

        ndata = slice.data.mapseq(func,type_in=slice.type,type_out=node.type,
                                  res_type=node.type,op=node.funcname, **node.kwargs)
        return slice.modify(data=ndata,rtype=node.type)
    
    def visitNoneToMissingOp(self, node, slice):
        ndata = slice.data.mapseq(none_to_missing,res_type=node.type, stype=node.type)
        return slice.modify(data=ndata)

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
        elif(node.packdepth == 1):
            ndata = ndata.pack(slice.type)
        else: #packdepth == 0
            ndata = ndata
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
                                       typeo=node.type,res_type=node.type,op=node.funcname)

        return slices[0].modify(data=ndata,name=node.name,rtype=node.type,dims=node.dims,bookmarks=node.bookmarks)

    def visitEquiJoinIndexOp(self, node, slices):
        ndatas = nested_array.co_map(joinindex,[slice.data for slice in slices],
                                       jointype=node.jointype,
                                       res_type=(node.results[0].type,node.results[1].type))
        return slices[0].modify(data=ndatas,name=None,rtype=None,dims=None, bookmarks=None)

    def visitSelectOp(self, node, slice):
        return slice.modify(data=slice.data[node.index],name=node.name, rtype=node.type, dims=node.dims, bookmarks=node.bookmarks)

    def visitFixateOp(self,node,slices=[]):
        res = []
        for cur_slice, slice in zip(node.sources, slices):
            ndata = slice.data.getStructuredData()
            nslice = slice.modify(data=ndata)
            nslice.setSource(cur_slice)
            res.append(nslice)
        return res
    
    def visitGatherOp(self,node,slices):
        res = []
        for cur_slice, slice in zip(node.sources, slices):
            slice.setSource(cur_slice)
            res.append(slice)
        return res
   
    def castto_any(self,castname,node,slice):
        dtype = node.type.toNumpy()
        return slice.data.mapseq(lambda x:numpy.cast[dtype](x),res_type=node.type)
    
    def castto_pickle(self,castname,node,slice):
        dtype = node.type.toNumpy()
        return slice.data.map(lambda x:cPickle.dumps(x,2),res_type=node.type)
    
    def castfrom_pickle(self,castname,node,slice):
        dtype = node.type.toNumpy()
        return slice.data.map(lambda x:cPickle.loads(str(x)),res_type=node.type)

    def castto_numbers(self,castname,node,slice):
        dtype = node.type.toNumpy()
        if not node.type.has_missing:
            return slice.data.mapseq(lambda x:numpy.cast[dtype](x),res_type=node.type)
        else:
            raise RuntimeError, "Casting missing values not yet supported"

    def caststring_to_int(self, castname, node, slice):
        if(node.type.has_missing):
            func = string_to_int_missing
        else:
            func = string_to_int
        dtype = node.type.toNumpy()

        return slice.data.mapseq(lambda x: func(x,dtype),res_type=node.type)
    
    def caststring_to_string(self, castname, node, slice):
        return slice.data

    def castto_array(self, castname, node, slice):
        func = stringset_to_array
        dtype = node.type.toNumpy()
        return slice.data.mapseq(func, dtype=dtype, res_type=node.type)
    
    
    def castto_bytes(self, castname, node, slice):
        if(node.type.has_missing):
            func = any_tobytes_missing
        else:
            func = any_tobytes
        return slice.data.mapseq(func, res_type=node.type)

    def caststring_to_real(self, castname, node, slice):
        if(node.type.has_missing):
            func = string_to_real_missing
        else:
            func = string_to_real
        dtype = node.type.toNumpy()

        return slice.data.mapseq(lambda x: func(x,dtype),res_type=node.type)
       

    def withinWithin(self, data, type1, type2, typeo, op, bcdepth=1):
        data1,data2 = data
        if(bcdepth > 1):
            data2 = util.darray([set(elem) for elem in data2.ravel()]).reshape(data2.shape)
            zipdata = numpy.broadcast(data1,data2)
        else:
            zipdata = zip(data1,data2)
        res = [elem in arr for elem,arr in zipdata]
        return util.darray(res,bool)

    def withinContains(self, data, type1, type2, typeo, op, bcdepth=1):
        data2,data1 = data
        if(bcdepth > 1):
            data2 = util.darray([set(elem) for elem in data2.ravel()]).reshape(data2.shape)
            zipdata = numpy.broadcast(data1,data2)
        else:
            zipdata = zip(data1,data2)
        res = [elem in arr for elem,arr in zipdata]
        return util.darray(res,bool)
    
    def mergeMerge(self, data, type1, type2, typeo, op):
        data1, data2 = data
        res = []
        for lelem, relem in zip(data1, data2):
            if lelem is relem:
                res.append(lelem)
            elif lelem is Missing:
                res.append(relem)
            elif relem is Missing:
                res.append(lelem)
            elif lelem == relem:
                res.append(lelem)
            else:
                raise RuntimeError, "Found unequal values during merge: " + str(lelem) + " != " + str(relem)
        return util.darray(res, typeo.toNumpy())

    def simple_arithDivide(self, data, type1, type2, typeo, op):
        data1,data2 = data
        if(data1 is Missing or data2 is Missing):
            return Missing
        filter = (data2 == 0) & (data1 == 0)
        if filter.any():
            data2 = data2.copy()
            data2[filter] = 1.0
        if util.numpy16up:
            res = numpy_arith[op](data1, data2, dtype=typeo.toNumpy())
        else:
            res = numpy_arith[op](data1, data2, sig=typeo.toNumpy())
        return res

    def simple_arithGeneral(self, data, type1, type2, typeo, op):
        data1,data2 = data
        if(data1 is Missing or data2 is Missing):
            return Missing
        if util.numpy16up:
            return numpy_arith[op](data1, data2, dtype=typeo.toNumpy())
        else:
            return numpy_arith[op](data1, data2, sig=typeo.toNumpy())
    boolboolGeneral = simple_arithGeneral

    def string_add_stringAdd(self, data, type1, type2, typeo, op):
        data1,data2 = data
        if(data1 is Missing or data2 is Missing):
            return Missing
        return util.darray(list(numpy_arith[op](numpy.cast[object](data1), numpy.cast[object](data2))),typeo.toNumpy())
    
    def array_add_arrayAdd(self, data, type1, type2, typeo, op):
        data1,data2 = data
        if(data1 is Missing or data2 is Missing):
            return Missing
        res = []
        dtype = typeo.toNumpy()
        for lelem, relem in zip(data1,data2):
            mshape = max(1,min(len(getattr(lelem,'shape',(0,))), len(getattr(relem,'shape',(0,)))))
            res.append(numpy.concatenate([util.darray(list(lelem),dtype,mshape),util.darray(list(relem),dtype,mshape)],axis=0))
        return util.darray(res,object)

    def simple_cmpGeneral(self, data, type1, type2, typeo, op):
        #a numpy bug gives all true arrays when using
        #bool as outtype in comparison
        return numpy_cmp[op](data[0], data[1])

    def string_cmpGeneral(self, data, type1, type2, typeo, op):
        #a numpy bug gives NotImplemented when performing operations,
        #such as "numpy.equal" on string arrays
        #so use direct operations ("__eq__")
        op = python_op[op]
        #Note: numpy seems to segfault, when using reverse operations on string comparision
        #using as first argument an  object array and as second a string array. 
        if(isinstance(data[0],numpy.ndarray) and data[0].dtype == object and isinstance(data[1],numpy.ndarray) and data[1].dtype != object):
            data = (data[0], numpy.cast[object](data[1]))

        res = getattr(data[0], op)(data[1])
        if(res is NotImplemented):
            if(isinstance(data[0],numpy.ndarray) and data[0].dtype != object and isinstance(data[1],numpy.ndarray) and data[1].dtype == object):
                data = (numpy.cast[object](data[0]), data[1])
            res = getattr(data[1], reverse_op[op])(data[0])
        assert not res is NotImplemented, "Not implemented error in stringstringGeneral for " \
                                            + str(op) + " and " + str(type1) + ", " + str(type2)
        return res

    def numberGeneral(self, data, type_in, type_out, op):
        if util.numpy16up:
            return numpy_unary_arith[op](data, dtype=type_out.toNumpy())
        else:
            return numpy_unary_arith[op](data, sig=type_out.toNumpy())
    
    def boolInvert(self, data, type_in, type_out, op):
        res = []
        for elem in data:
            if elem is Missing:
                res.append(elem)
            else:
                res.append(not elem)
        return util.darray(res, type_out.toNumpy())

    def repmissingReplaceMissing(self, data, type_in, type_out, op, def_value=NOVAL):
        if not type_in.has_missing:
            return data
        if def_value is NOVAL:
            if type_out.__class__ is rtypes.TypeArray and type_out.dims[0].shape == UNDEFINED:
                for elem in data:
                    if not elem is Missing:
                        shape = len(elem)
                        break
                else:
                    raise RuntimeError, "Could not determine shape of dimension. ReplaceMissing impossible. Change to variable dim."
               
                dim = dimensions.Dim(shape, type_out.dims[0].dependent)
                def_value = dimpaths.dimsToArrays(dimpaths.DimPath(dim),type_out.subtypes[0]).toDefval()
            else:                
                def_value = type_out.toDefval()
        res = []
        for elem in data:
            if elem is Missing:
                res.append(def_value)
            else:
                res.append(elem)
        return util.darray(res, type_out.toNumpy())
    
    def ismissingIsMissing(self, data, type_in, type_out, op, def_value=NOVAL):
        res = []
        for elem in data:
            if elem is Missing:
                res.append(True)
            else:
                res.append(False)
        return util.darray(res, type_out.toNumpy())
   
    def corrCorr(self, data, type_in, type_out, op):
        intype = type_in.toNumpy()
        res = []
        for elem in data:
            if(len(elem.shape) < 2):
                elem = util.darray(list(elem),intype,2,2)
            res.append(numpy.corrcoef(elem))
        return util.darray(res,object,1,1)

    def eachEach(self, data, type_in, type_out, op, eachfunc):
        if(isinstance(eachfunc,context.Context)):
            return util.darray([context._apply(eachfunc,elem) for elem in data],type_out.toNumpy())
        else:
            return util.darray([eachfunc(elem) for elem in data],type_out.toNumpy())

    def sortableArgsort(self, data, type_in, type_out, op, packdepth, descend=False):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        if(len(data.shape) < 2):
            if(descend):
                res = util.darray([numpy.flipud(numpy.argsort(row,axis=0)) for row in data],object)
            else:
                res = util.darray([numpy.argsort(row,axis=0) for row in data],object)
        else:
            res = numpy.argsort(data,axis=1)
            if(descend):
                res = res[:,::-1,...]
        return res
    
    def sortableRank(self, data, type_in, type_out, op, packdepth, descend=False):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        if(len(data.shape) < 2):
            if(descend):
                res = util.darray([numpy.argsort(numpy.flipud(numpy.argsort(row,axis=0)),axis=0) for row in data],object)
            else:
                res = util.darray([numpy.argsort(numpy.argsort(row,axis=0),axis=0) for row in data],object)
        else:
            res = numpy.argsort(data,axis=1)
            if(descend):
                res = res[:,::-1,...]
            res = numpy.argsort(res,axis=1)                
        return res
    
    def fixdimCumSum(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        if(len(data.shape) < 2):
            res = util.darray([numpy.cumsum(row,axis=0) for row in data],object)
        else:
            res = numpy.cumsum(data,axis=1)
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
                            subres.append(util.darray([r] * len(d),dtype))
                        res = util.darray(subres,object)
                    xres.append(res)
            xres = util.darray(xres)
        elif(len(data.shape) == 2):
            xres = numpy.tile(numpy.arange(data.shape[1],dtype=dtype),data.shape[0]).reshape(data.shape[:2])
            if(packdepth > 1):
                res = []
                for r,d in zip(xres.ravel(),data.ravel()):
                    res.append(util.darray([r] * len(d),dtype))
                xres = util.darray(res,object,1,1).reshape(data.shape)
        else:
            assert len(data.shape) == 3, "Unexpected data shape"
            r =  numpy.tile(numpy.arange(data.shape[1],dtype=dtype),data.shape[0]).reshape(data.shape[:2])
            xres = numpy.repeat(r,data.shape[2]).reshape(data.shape)
           
        return xres

    def countCount(self, data, type_in, type_out, op, packdepth):
        dtype = type_out.toNumpy()
        if(len(data.shape) == 1):
            return util.darray([len(row) for row in data],dtype)
        else:
            return util.darray([data.shape[1]] * data.shape[0],dtype)
    
    def setGeneral(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        dtype = type_out.toNumpy()
        if(packdepth > 1):
           return util.darray([[set(subrow) for subrow in row.transpose()] for row in data],dtype,2,2)
        else:
           return util.darray([set(row) for row in data],dtype)
    
    def uniqueGeneral(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        dtype = type_out.toNumpy()
        if(packdepth > 1):
           return util.darray([[numpy.unique(subrow,return_index=True)[1] for subrow in row.transpose()] for row in data],dtype,2,2)
        else:
           return util.darray([numpy.unique(row, return_index=True)[1] for row in data],dtype)
   
    def arrayarraySum(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        if(packdepth > 1):
            return util.darray([[numpy.concatenate(list(subrow),axis=0) for subrow in row.transpose()] for row in data],object,2,2)
        else:
            return util.darray([numpy.concatenate(list(row),axis=0) for row in data],object)
    
    def stringstringSum(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        if(packdepth > 1):
            return util.darray([[''.join(list(subrow)) for subrow in row.transpose()] for row in data],object,2,2)
        else:
            return util.darray([''.join(list(row)) for row in data],object)
    
    def fixdimArgmax(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        func = numpy_dimfuncs[op]
        dtype = type_out.toNumpy()
        if type_in.has_missing:               
            res = []
            for row in data:
                pos = numpy.arange(len(row))
                filter = ~numpy.equal(row, Missing)
                pos = pos[filter]
                row = row[filter]
                res.append(pos[func(row,axis=0)])
            return util.darray(res,dtype)
        else:            
            if(len(data.shape) < 2):
                if(packdepth > 1):
                    dtype = object
                return util.darray([func(row,axis=0) for row in data],dtype)
            else:
                return func(data,axis=1)
    fixdimArgmin = fixdimArgmax

    def fixdimGeneral(self, data, type_in, type_out, op, packdepth):
        data = ensure_fixeddims(data,packdepth,type_in.toNumpy())
        func = numpy_dimfuncs[op]
        dtype = type_out.toNumpy()
        if type_in.has_missing:                
            return util.darray([func([elem for elem in row if not elem is Missing],axis=0) for row in data],dtype)
        else:            
            if(len(data.shape) < 2):
                if(packdepth > 1):
                    dtype = object
                return util.darray([func(row,axis=0) for row in data],dtype)
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
    'Argmin':numpy.argmin,
    'Argmax':numpy.argmax,
    'Std':numpy.std
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
    nseq = util.darray(zip(*seqs))
    return nseq


def speedarrayify(seqs,dtype):
    nseq = numpy.array(seqs,dtype).T
    return nseq


def speeddictindex(seqs,dtype, allow_missing=False, keep_missing=False):
    if keep_missing:
        return util.darray([d.get(k,k) for d, k in zip(*seqs)],dtype)
    elif allow_missing:
        return util.darray([d.get(k,Missing) for d, k in zip(*seqs)],dtype)
    else:
        return util.darray([d[k] for d, k in zip(*seqs)],dtype)


def speedfilter(seqs,has_missing, ctype, stype):
    data,constraint = seqs
    if data is Missing:
        return data
    if len(data) == 0: #FIXME
        return data
    if(has_missing):
        if(isinstance(ctype,rtypes.TypeArray)):
            missing = stype.subtypes[0].toMissingval()
            if(isinstance(ctype.subtypes[0],rtypes.TypeBool)):
                res = []
                for pos, elem in enumerate(constraint.ravel()):
                    if(elem is Missing):
                        res.append(missing)
                    elif(elem == True):
                        res.append(data[pos])
                res = util.darray(res,object)
            else:#indices
                res = []
                for elem in constraint.ravel():
                    if(elem is Missing):
                        res.append(missing)
                    else:
                        res.append(data[elem])
                res = util.darray(res,object)
        else:
            if(constraint is Missing):
                missing = stype.toMissingval()
                res = missing
            else:
                try:
                    res = data[constraint]
                except Exception:
                    res = util.darray(data)[constraint]
    else:
        try:
            res = data[constraint]
        except Exception:
            res = util.darray(data)[constraint]
    return res

def ensure_fixeddims(seqs,packdepth,dtype):
    if(packdepth > 1):
        if(len(seqs.shape) == 1):
            res = []
            for seq in seqs:
                if(len(seq.shape) >= 2):
                    res.append(seq)
                else:
                    res.append(util.darray(seq,dtype,100000,2))
            res = util.darray(res,object)
        elif(len(seqs.shape) == 2):
            res = util.darray(seqs.tolist(),dtype,1000,3)
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
        indexdata = [util.darray([],int)] * numpy.prod(shape)
        indexdata = util.darray(indexdata, object, 1).reshape(tuple(shape))
        
        for key, posses in data_dict.iteritems():
            loc = tuple([index[keypart] for index, keypart in zip(indexes,key)])
            indexdata[loc] = util.darray(posses, int)
    else:
        data_dict = defaultdict(list)
        for pos, elems in enumerate(*data):
            if(elems is Missing):
                data_dict["__TEMP__" + str(elems.__hash__())].append(pos)
            else:
                data_dict[elems].append(pos)
        indexdata = [util.darray(elem, int) for elem in data_dict.values()]
        indexdata = util.darray(indexdata, object, 1)
   
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

    return (util.darray(tlpos,ldtype), util.darray(trpos,rdtype))

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


def stringset_to_array(seq, dtype):
    res = []
    for elem in seq:
        if elem is Missing:
            res.append(Missing)
        else:
            res.append(util.darray(list(elem),dtype))
    return util.darray(res)


def any_tobytes_missing(seq):
    res = []
    for elem in seq:
        if elem is Missing:
            res.append(Missing)
        else:
            res.append(str(elem))
    return util.darray(res)

def any_tobytes(seq):
    res = [str(elem) for elem in seq]
    return util.darray(res)


def string_to_int_missing(seq, dtype):
    res = []
    for elem in seq:
        if elem is Missing or elem == "":
            res.append(Missing)
        else:
            try:
                res.append(int(elem))
            except ValueError:
                res.append(Missing)
    return util.darray(res,dtype)

def string_to_int(seq, dtype):
    return util.darray([int(elem) for elem in seq],dtype)


def string_to_real_missing(seq, dtype):
    res = []
    for elem in seq:
        if elem is Missing or elem == "":
            res.append(Missing)
        else:
            try:
                res.append(float(elem))
            except ValueError:
                res.append(Missing)
    return util.darray(res,dtype)

def string_to_real(seq, dtype):
    return util.darray([float(elem) for elem in seq],dtype)


def none_to_missing(seq,stype):
    missing = stype.toMissingval()
    seq = seq.copy()
    seq[numpy.equal(seq,None)] = missing
    return seq
   
def detect_shape(data, stype):
    if stype.__class__ is rtypes.TypeArray:
        if not stype.dims[0].dependent and stype.dims[0].shape == UNDEFINED:
            stype.dims[0].shape = len(data)

        subtype = detect_shape(data[0],stype.subtypes[0])
        stype = stype.copy()
        stype.subtypes= (subtype,)
    
    return stype

