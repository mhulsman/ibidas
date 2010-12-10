import operator
from itertools import chain
from collections import defaultdict
import numpy


import wrapper
from ..constants import *
from ..utils.multi_visitor import VisitorFactory, F_CACHE, NF_ERROR, NF_ELSE
from ..itypes import rtypes
from ..passes import required
from .. import engines

_delay_import_(globals(),"..representor")
_delay_import_(globals(),"..utils","util","nestutils","cutils")
_delay_import_(globals(),"..slices")
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
        data_slice = slices.UnpackArraySlice(data_slice)

    res = wrapper.SourceRepresentor((data_slice,)) 

    if(unpack and isinstance(res.getType(), rtypes.TypeTuple)):
        res = repops_slice.unpack_tuple(res)
    
    return res#}}}



def var(name, example_data=None, dtype=rtypes.unknown, 
        type_check=False, unpack=True):#{{{
    if(isinstance(dtype, str)):
        dtype = rtypes.createType(dtype)
    if(isinstance(example_data, representor.Representor)):
        assert dtype == rtypes.unknown, \
            "Example representor and dtype cannot both be supplied"
        res = example_data.copy()
    else:
        res = rep(example_data, dtype, type_check=type_check, unpack=unpack)
    res.__class__ = VarPyRepresentor
    res._name = name
    res._type_check = type_check
    del res._data
    return res#}}}


class PySelectIndicator(wrapper.Indicator):
    pass

class Result(object):
    __slots__ = ['data', 'ntypes', 'ndims']
    def __init__(self, data=None, ntypes=None, ndims=None):
        if(ntypes is None):
            self.ntypes = {}
        else:
            assert isinstance(ntypes, dict), "Ntypes attribute should be a dict"
            self.ntypes = ntypes

        if(ndims is None):
            self.ndims = {}
        else:
            assert isinstance(ndims, dict), "Ndims attribute should be a dict"
            self.ndims = ndims

        if(data is None):
            data = {}
        else:
            assert isinstance(data, dict), "Data attribute should be dict"
            self.data = data

    def copy(self):
        return Result(self.data.copy(), self.ntypes.copy(), self.ndims.copy())

    def __repr__(self):
        return "\nRESULT\nData: " + str(self.data) + \
                       "\nNtypes:   " + str(self.ntypes) + \
                       "\nNdims:   " + str(self.ndims) + "\n"

def merge_results(*results):
    ndata = {}
    ntypes = {}
    ndims = {}
    for result in results:
        ndata.update(result.data)
        ntypes.update(result.ntypes)
        ndims.update(result.ndims)
    return Result(ndata, ntypes, ndims)

class PySelectExecutor(VisitorFactory(prefixes=("rep", ), flags=F_CACHE),
                    VisitorFactory(prefixes=("sliceop", "op"), flags=NF_ELSE),
                    wrapper.Executor):
    after = set([required.RequiredSliceIds])

    @classmethod
    def run(cls, query, pass_results):
        self = cls()
        self.query = query
        #pylint: disable-msg=E1101
        return (self.rep(query.root),)
    
    def getSliceById(self, sid, crep, res):
        slice = crep._all_slices[sid]
        if(not sid in res.data):
            if(slice.last_id in res.data):
                res.data[sid] = res.data[slice.last_id]
                if(slice.last_id in res.ntypes):
                    res.ntypes[sid] = res.ntypes[slice.last_id]
                if(slice.last_id in res.ndims):
                    res.ndims[sid] = res.ndims[slice.last_id]
            else:
                #pylint: disable-msg=E1101
                (ndata, ntype, ndims) = self.sliceop(slice, crep, res)
                res.data[slice.id] = ndata
                if(slice.type.__class__ is rtypes.TypeUnknown and
                    not ntype.__class__ is rtypes.TypeUnknown):
                    res.ntypes[slice.id] = ntype
                    res.ndims[slice.id] = ndims
 
        ftype = res.ntypes.get(sid, slice.type)
        fdims = res.ndims.get(sid, slice.dims)
        return (res.data[sid], ftype, fdims)

    def repRepresentor(self, crep):
        #pylint: disable-msg=E1101
        result = self.op(crep).copy()
        
        for rid in crep._req_ids:
            #inlining of getSliceByID
            if(rid in result.data):
                continue
            slice = crep._all_slices[rid]
            if(slice.last_id in result.data):
                result.data[rid] = result.data[slice.last_id]
                if(slice.last_id in result.ntypes):
                    result.ntypes[rid] = result.ntypes[slice.last_id]
                if(slice.last_id in result.ndims):
                    result.ndims[rid] = result.ndims[slice.last_id]
                continue
            #pylint: disable-msg=E1101
            (ndata, ntype, ndims) = self.sliceop(slice, crep, result)
            result.data[slice.id] = ndata
            if(slice.type.__class__ is rtypes.TypeUnknown and
                not ntype.__class__ is rtypes.TypeUnknown):
                result.ntypes[slice.id] = ntype
                result.ndims[slice.id] = ndims
        
        #some memory saving (not necessary)
        for did in result.data.keys():
            if(not did in crep._req_ids):
                del result.data[did]
        return result 
    
    def repfixate(self, crep):
        #pylint: disable-msg=E1101
        res = self.op(crep)
        xres = {}
        for slice in crep._active_slices:
            xres[slice.id] = res.data[slice.id]
        return xres

    def opPyRepresentor(self, crep):
        return Result(crep._data.copy())
    
    def opVarPyRepresentor(self, crep):
        assert crep._name in self.query.args, \
            "Variable " + str(crep._name) + " not found in arguments."
        data = self.query.args[crep._name]
        if(isinstance(data, representor.Representor)):
            if(not crep._no_type_check):
                assert data.type <= crep.type, "Argument has incompatible type"
        else:
            data = rep(data, crep.type, no_type_check=crep._no_type_check)

        res = {}
        for vslice, aslice in zip(crep._active_slices, data._active_slices):
            res[vslice.id] = data._data[aslice.id]

        return Result(res)
    
    def opUnaryOpRep(self, crep):
        #pylint: disable-msg=E1101
        return self.rep(crep._sources[0])

    def opMultiOpRep(self, crep):
        return merge_results(*[self.rep(s) for s in crep._sources])
    
    def opRFilter(self, crep):
        result = self.rep(crep._sources[0]).copy()
        for source, cslice, fslices, fdepth, fbcast in \
                             zip(crep._sources[1:], crep._constraint_slices, 
                                   crep._filter_slices, crep._filter_depths, crep._filter_broadcast):
            constraint_res = self.rep(source)
            cdata = constraint_res.data[cslice.id]

            for depth, slice in zip(fdepth, fslices):
                if(slice.id in crep._req_ids):
                    rdata = result.data[slice.id]
                    for d in depth:
                        if(d):
                            nfbcast = ((1,) * d) + fbcast
                        else:
                            nfbcast = fbcast
                        rdata = nestutils.nest_broadcast(nfbcast, [rdata, cdata], [slice.dims, cslice.dims], 
                                                        inner_filter, (slice.type,), slice.dims, vectorize=1 )
                    result.data[slice.id] = rdata
        return result

    def opGroup(self, crep):
        result = self.rep(crep._sources[0]).copy()
        gresult = self.rep(crep._sources[1]) 
       

        gdata = [gresult.data[slice.id] for slice in crep._group_slices]
        gdims = [slice.dims for slice in crep._group_slices]
        bcast = [1] * (len(gdims[0]) - 1)
        groupdims = crep._keep_slices["__group__"][2]
        gidx = nestutils.nest_broadcast(bcast, gdata, gdims, inner_group, (rtypes.TypeInt64(),), groupdims)
        
        nresult = {}
        for id in crep._req_ids:
            nresult[id] = result.data[id]

        args = {'count_sel':None}
        for key, value in crep._keep_slices.iteritems():
            if(key == "__group__"):
                nbcast = (3,) * (len(gdims[0]) - 1) + (2,) * len(gdata)
                gindex = gidx
            else:
                nbcast = (3,) * (len(gdims[0]) - 1) + (2,) * len(key)
                args['key'] = key
                gindex = nestutils.nest_broadcast((1,) * (len(gdims[0]) - 1), (gidx,), (groupdims,), inner_groupcondense, (rtypes.TypeInt64(),), groupdims[:(len(gdims[0]) - 1 + len(key))], func_attr=args)
                
            xslices = value[0]
            depths = value[1]
            xdims = value[2]

            for depth, slice in zip(depths, xslices):
                if(slice.id in crep._req_ids):
                    rdata = nresult[slice.id]
                    for d in depth:
                        if(d):
                            nfbcast = ((1,) * d) + nbcast
                        else:
                            nfbcast = nbcast
                        rdata = nestutils.nest_broadcast(nfbcast, [rdata, gindex], [slice.dims, xdims], 
                                                        inner_filter, (slice.type,), slice.dims, vectorize=1)
                    nresult[slice.id] = rdata

        

        return Result(nresult)
   
    def opflat(self, crep):
        result = self.rep(crep._sources[0]).copy()
       
        sdepths = []
        sdatas = []
        stypes = []
        sids = []
        for sliceid in crep.inner_req_ids:
            slice = crep._all_slices[sliceid]
            sdim = slice.dims
            depth = 0
            for sd, fd in zip(sdim, crep._flat_dim):
                if(sd == fd):
                    depth += 1
                else:
                    break
            if(sliceid in crep._flat_slices):
                depth += 1
            if(depth >= len(crep._flat_dim) - 1):
                sdepths.append(depth)
                sdatas.append(result.data[sliceid])
                if(len(sdim) >= len(crep._flat_dim)):
                    stypes.append(object)
                else:
                    stypes.append(slice.type.toNumpy())
                sids.append(sliceid)
        nsdatas = nestutils.nest_flatten(sdatas, sdepths, stypes)
        for nsdata, sliceid in zip(nsdatas, sids):
            result.data[sliceid] = nsdata
        
        return result

    def opJoin(self, crep):
        s1 = self.rep(crep._sources[0]).copy()
        s2 = self.rep(crep._sources[1]).copy()
        (lsliceids, rsliceids) = crep._join_slices

        
        #determine fields to be joined
        lids = lsliceids & crep._req_ids
        rids = rsliceids & crep._req_ids

        #check if there is at least one field, otherwise add it
        if(not lids):
            lids = [(lsliceids.intersection(s1.data.keys())).pop()]
        if(not rids):
            rids = [(rsliceids.intersection(s2.data.keys())).pop()]
        
        #ensure constant order
        lids = list(lids)
        rids = list(rids)

        #get data to be joined
        ldata = [s1.data[sliceid] for sliceid in lids]
        rdata = [s2.data[sliceid] for sliceid in rids]
         
        res = nestutils.nest_crossjoin(crep._join_broadcast, ldata, rdata, crep._join_group)        
        result = merge_results(s1, s2)
        
        for id, col in zip(chain(lids, rids), res):
            result.data[id] = col
        
        return result

    def opEquiJoin(self, crep):
        s1 = self.rep(crep._sources[0]).copy()
        s2 = self.rep(crep._sources[1]).copy()
        (lsliceids, rsliceids) = crep._join_slices
        
        #determine fields to be joined
        lids = lsliceids & crep._req_ids
        rids = rsliceids & crep._req_ids
        
        #check if there is at least one field, otherwise add it
        if(not lids):
            lids = [(lsliceids.intersection(s1.data.keys())).pop()]
        if(not rids):
            rids = [(rsliceids.intersection(s2.data.keys())).pop()]

        #ensure constant order
        lids = list(lids)
        rids = list(rids)

        #get data to be joined
        ldata = [s1.data[sliceid] for sliceid in lids]
        rdata = [s2.data[sliceid] for sliceid in rids]

        cldata = s1.data[crep._lsliceid]
        crdata = s2.data[crep._rsliceid]
        cldepth = len(crep._sources[0]._all_slices[crep._lsliceid].dims)
        crdepth = len(crep._sources[1]._all_slices[crep._rsliceid].dims)

        res = nestutils.nest_equijoin(crep._join_broadcast, ldata, rdata, cldata, crdata, cldepth, crdepth, crep._join_group)        
        result = merge_results(s1, s2)
        
        for id, col in zip(chain(lids, rids), res):
            result.data[id] = col
        
        return result
       

    def opelse(self, crep):
        assert hasattr(crep, "pyexec"), \
                "Cannot find execution method for " + str(crep.__class__)
        return crep.pyexec(self)
   
    def sliceopSlice(self, slice, crep, res):
        raise RuntimeError, "Cannot find source for slice " + \
                                str(slice) + " (id:" + str(slice.id) + ")"

    def sliceopUnpackArraySlice(self, cslice, crep, res):
        sdata, stype, sdims = self.getSliceById(cslice.source_ids[0], crep, res)
        assert isinstance(stype, rtypes.TypeArray), \
            "Unpacking array failed: source is not an array but " + str(stype)

        subtype = stype.subtypes[0]
        dconvertor = convertors.getConvertor(subtype)
        ndims = sdims + stype.dims
        func = lambda data, bcast, dims, pack, rtypes, rdims, args: dconvertor.convert(rtypes[0], data[0]) 
        ndata = nestutils.nest_broadcast([1] * len(sdims), [sdata], [sdims], func, [subtype], ndims)
        return (ndata, subtype, ndims)
    
        
    def sliceopUnpackTupleSlice(self, cslice, crep, res):
        sdata, stype, sdims = self.getSliceById(cslice.source_ids[0], crep, res)
        assert isinstance(stype, rtypes.TypeTuple), \
            "Unpacking tuple failed: source is not an tuple but " + str(stype)
        
        subtype = stype.subtypes[cslice.slice_idx]
        scanner = scanner_protocol.getScanner(subtype)

        inner_func = operator.itemgetter(cslice.slice_idx)
        if(stype.has_missing):
            inner_func = util.filter_missing(inner_func)
        ndata = nestutils.nest_broadcast([1] * (len(sdims) - 1), [sdata], [sdims], inner_detuple, 
                                    [subtype], sdims, func_attr={'selector':inner_func, 'scanner':scanner})
        return (ndata, subtype, sdims)
        

    def sliceopBinElemOpSlice(self, cslice, crep, res):
        data1, type1, dims1 = self.getSliceById(cslice.source_ids[0], crep, res)
        data2, type2, dims2 = self.getSliceById(cslice.source_ids[1], crep, res)
        xfunc = cslice.oper.exec_funcs["py"]
        func = lambda x, y: xfunc(x, y, type1, type2, cslice.type, cslice.op)
        rdata = nestutils.nestop([data1, data2], func, [len(dims1), len(dims2)])
        return (rdata, cslice.type, cslice.dims)
    
    def sliceopUnaryElemOpSlice(self, cslice, crep, res):
        sdata, stype, sdims = self.getSliceById(cslice.source_ids[0], crep, res)
        func = cslice.oper.exec_funcs["py"]
        xfunc = lambda data, bcast, dim, pack, rtypes, rdims, args: func(data[0], stype, rtypes[0], cslice.op) 
        ndata = nestutils.nest_broadcast([1] * (len(sdims) -1), [sdata], [sdims], xfunc, [cslice.type], cslice.dims, vectorize=1)
        return (ndata, cslice.type, cslice.dims)
       
    def sliceopMapSeqSlice(self, cslice, crep, res):
        sdata, stype, sdims = self.getSliceById(cslice.source_ids[0], crep, res)
        if(cslice.params or cslice.kwds):
            efunc = cslice.exec_func
            params = cslice.params
            kwds = cslice.kwds
            xfunc = lambda x : efunc(x, *params, **kwds)
        else:
            xfunc = cslice.exec_func
        rdata = nestutils.nestmap(sdata, xfunc, len(sdims) - 1)
        return (rdata, cslice.type, cslice.dims)
    
    def sliceopMapSlice(self, cslice, crep, res):
        sdata, stype, sdims = self.getSliceById(cslice.source_ids[0], crep, res)
        xfunc = cslice.exec_func
        rdata = nestutils.nestmap(sdata, xfunc, len(sdims), cslice.type.toNumpy())
        return (rdata, cslice.type, cslice.dims)
  
    def sliceopPackTupleSlice(self, cslice, crep, res):
        data = []
        for sourceid in cslice.source_ids:
            sdata, stype, sdims = self.getSliceById(sourceid, crep, res)
            if(cslice.to_python):
                sdata = sdata.tolist()
            data.append(sdata)
        rfunc = lambda *x: tuple(x)
        rdata = nestutils.nestop(data, rfunc, [len(cslice.dims) + 1] * len(data)) 
        return (rdata, cslice.type, cslice.dims)


    def sliceopPackArraySlice(self, cslice, crep, res):
        sdata, stype, sdims = self.getSliceById(cslice.source_ids[0], crep, res)
        return (sdata, cslice.type, cslice.dims)
    
    def sliceopPackListSlice(self, cslice, crep, res):
        sdata, stype, sdims = self.getSliceById(cslice.source_ids[0], crep, res)
        sdata = nestutils.nestmap(sdata, lambda x: x.tolist(), len(cslice.dims))
        return (sdata, cslice.type, cslice.dims)

    def sliceopFreezeSlice(self, cslice, crep, res):
        sdata, stype, sdims = self.getSliceById(cslice.source_ids[0], crep, res)

        scanner = scanner_protocol.getScanner(stype)
        if(sdims):
            rdata = data_freezer.freeze(scanner, sdata, stype)
        else:
            rdata = data_freezer.freeze(scanner, [sdata], stype)[0]
      
        return (rdata, cslice.type, cslice.dims)
      
    def sliceopAggregateSlice(self, cslice, crep, res):
        sdata, stype, sdims = self.getSliceById(cslice.source_ids[0], crep, res)
        xfunc = cslice.exec_func
        rdata = nestutils.nestmap(sdata, xfunc, len(sdims) - 1,
                                        cslice.type.toNumpy())
        return (rdata, cslice.type, cslice.dims[:-1])
   

#helper funcs
def inner_detuple(data, bcast, dims, pack, rtypes, rdims, args):
    selector = args['selector']
    if(not rdims):
        return data_convertor.convert(args['scanner'], [selector(data[0])], rtypes[0])[0]
    else:
        return data_convertor.convert(args['scanner'], [selector(elem) for elem in data[0]], rtypes[0]) 


def inner_group(data, bcast, dims, pack, rtypes, rdims, args):
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
            data_dict[elems].append(pos)

        indexdata = [cutils.darray(elem, int) for elem in data_dict.values()]
        indexdata = cutils.darray(indexdata, object, 1)
   
    return indexdata


def inner_groupcondense(data, bcast, dims, pack, rtypes, rdims, args):
    ndim = len(dims[0]) -1
    key = args['key']
    firstitem =operator.itemgetter(0)
    if(ndim > 1):
        if(args['count_sel'] is None):
            args['count_sel'] = cutils.darray([len(elem) for elem in data[0].ravel()],int).reshape(data[0].shape)
        tidx = set(range(ndim))
        tidx = tidx - set(key)
        tidx = list(tidx)
        tidx.sort()
        csel = numpy.transpose(args['count_sel'], key + tuple(tidx))
        rshape = tuple([args['count_sel'].shape[ti] for ti in key]) +  (numpy.prod([args['count_sel'].shape[ti] for ti in tidx]),)
        csel = csel.reshape(rshape)
        idx = numpy.argmax(csel, axis=len(key))
        cgidx = numpy.transpose(data[0], key + tuple(tidx))
        cgidx = cgidx.reshape(rshape)

        other_idx = numpy.indices(rshape[:len(key)])                    
        cgidx = cgidx[tuple(other_idx) + (idx,)]
        gindex = cutils.darray([firstitem(elem) for elem in cgidx.ravel()],int).reshape(cgidx.shape)
    else:
        assert len(key) == 1 and key[0] == 0, "Invalid keep index"
        gindex = cutils.darray([elem[0] for elem in data[0]],int)
    return gindex
       
def inner_filter(data, bcast, dims, pack, rtypes, rdims, args):
    filter_data, constraint_data = data
    if(isinstance(constraint_data, numpy.ndarray) and constraint_data.dtype == object):
        constraint_data = numpy.cast[bool](constraint_data)
    if(len(bcast) > 0):
        constraint_data = (slice(None),) * len(bcast) + (constraint_data,)
    return filter_data[constraint_data]
