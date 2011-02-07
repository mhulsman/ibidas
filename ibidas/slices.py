import copy
from collections import defaultdict
from itertools import izip_longest

from constants import *
from utils import util
from itypes import rtypes,dimpaths
from query_graph import Node


_delay_import_(globals(),"itypes","dimensions","typeops","convertors","casts")
_delay_import_(globals(),"itypes.type_attribute_freeze","freeze_protocol")

#pylint: disable-msg=E1101
sliceid = util.seqgen().next
class Slice(Node):#{{{
    """A slice represents a attribute and set of dimensions in the data.
    Each slice has an id for unique identification, an attribute name
    for accessing it, a type describing the contents of the slice,
    a dims attribute describing the packaging of the attribute, 
    and a last_id, describing a slice from which the current
    slice has been copied (id's within a table should be unique).
    
    Note: one can have multiple slices with dissimilar dimensions
    but the same id. An id represents similarity of the content 
    on element level, dims the packaging."""

    __slots__ = ['name', 'type', 'dims','bookmarks']


    def __init__(self, name, rtype = rtypes.unknown, dims=dimpaths.DimPath(), bookmarks=set()):
        """Creates a slice object.

        :param name: name of slice (string)
        :param type: type of slice, optional, default = unknown
        :param dims: tuple of Dim objects, optional, default = ()
        """
        assert isinstance(name, (str,unicode)), "Name of slice should be a string"
        assert (name.lower() == name), "Name should be in lowercase"
        assert isinstance(rtype, rtypes.TypeUnknown), "Invalid type given"
        assert isinstance(dims, dimpaths.DimPath), "Dimensions of a slice should be a DimPath"
        assert isinstance(bookmarks,set), "Bookmarks should be a set"
        assert all([isinstance(bm,(str,unicode)) for bm in bookmarks]),"Bookmarks should be a string"
        assert all([bm.lower() == bm for bm in bookmarks]), "Bookmarks should be in lowercase"

        self.name = name
        self.type = rtype
        self.dims = dims
        self.bookmarks = bookmarks
    
    def __repr__(self):
        res = self.name
        res += "="
        if(self.dims):
            dimstr = []
            for pos, dim in enumerate(self.dims):
                dimstr.append("[" + str(dim) + "]")
            res += "<".join(dimstr) + "<"
                
        res += str(self.type)
        return res#}}}

class UnaryOpSlice(Slice):#{{{
    __slots__ = ['source']
    def __init__(self, source, name=None, rtype=None, dims=None, bookmarks=None):
        if(name is None):
            name = source.name
        if(rtype is None):
            rtype = source.type
        if(dims is None):
            dims = source.dims
        if(bookmarks is None):
            bookmarks = source.bookmarks
        self.source = source
        assert isinstance(source,Slice),"Source of UnaryOpSlice should be a slice"
        Slice.__init__(self, name, rtype, dims, bookmarks)#}}}

class MultiOpSlice(Slice):#{{{
    __slots__ = ['sources']
    def __init__(self, source_slices, name, rtype=rtypes.unknown, 
                            dims=dimpaths.DimPath(), bookmarks=set()):
        self.sources = tuple(source_slices)
        Slice.__init__(self, name, rtype, dims, bookmarks)#}}}

class LinkSlice(UnaryOpSlice):#{{{
    __slots__ = ['link']
    def __init__(self, source, link, name, rtype=rtypes.unknown, dims=dimpaths.DimPath(), bookmarks=set()):
        assert isinstance(link,representor.Representor),"Link of LinkSlice should be a representor"
        UnaryOpSlice.__init__(self, source, name, rtype, dims, bookmarks)#}}}

class ExtendSlice(Slice):#{{{
    __slots__ = []
#}}}

class DataSlice(Slice):#{{{
    __slots__ = ['data']
    def __init__(self, data, name=None, rtype=None, dims=dimpaths.DimPath(), bookmarks=set()):
        self.data = data
        Slice.__init__(self, name, rtype, dims, bookmarks)#}}}

class ChangeBookmarkSlice(UnaryOpSlice):#{{{
    __slots__ = []

    def __init__(self,source,add_bookmark=None,update_auto_bookmarks=None):
        nbookmarks = source.bookmarks.copy()

        if(not update_auto_bookmarks is None):
            for bm in nbookmarks:
                if(bm.startswith('!')):
                    nbookmarks.discard(bm)
                    nbookmarks.add("!" + update_auto_bookmarks + bm[1:])

        if(not add_bookmark is None):
            nbookmarks.add(add_bookmark)
        
        UnaryOpSlice.__init__(self, source, bookmarks=nbookmarks)#}}}

class ChangeNameSlice(UnaryOpSlice):#{{{
    __slots__ = []
    def __init__(self,source, new_name):
        UnaryOpSlice.__init__(self, source, name=new_name)#}}}

class ChangeDimPathSlice(UnaryOpSlice):#{{{
    __slots__ = []
    def __init__(self,source, new_dims):
        UnaryOpSlice.__init__(self, source, dims=new_dims)#}}}

class CastSlice(UnaryOpSlice):#{{{
    __slots__ = ["cast_name"]
    def __init__(self, source, new_type):
        cast_name = casts.canCast(source.type,new_type)
        assert not cast_name is False, "Cannot cast " + str(source.type) + " to " + str(new_type)
        self.cast_name = cast_name
        UnaryOpSlice.__init__(self, source, rtype=new_type)#}}}
        
class DetectTypeSlice(UnaryOpSlice):#{{{
    __slots__ = []#}}}

class UnpackArraySlice(UnaryOpSlice):#{{{
    """An slice which is the result of unpacking a source slice."""
    __slots__ = ["unpack_dims"]

    def __init__(self,slice,ndim=None):
        """Creates a new slice, and sets attributes.

        :param slice: Source slice to be unpacked"""
        stype = slice.type

        if(ndim is None):
            if(not stype.dims):
                ndim = 1
            else:
                ndim = len(stype.dims)
        
        unpack_dims = []
        rest_dims = []
        while ndim:
            assert isinstance(stype, rtypes.TypeArray), "Cannot unpack slice deep enough " + \
                                str(slice.name) + " as there is no array type"
            
            if(not stype.dims):
                nudims = (dimensions.Dim(UNDEFINED, variable=(True,) * (len(slice.dims) + len(unpack_dims)), 
                            has_missing=stype.has_missing),)
            else: 
                nudims = stype.dims[:ndim]
                rest_dims.extend(stype.dims[ndim:])
            last_type = stype
            stype = stype.subtypes[0]
            ndim -= len(nudims)
            unpack_dims.extend(nudims)
        
        if(rest_dims):
            ntype = rtypes.TypeArray(last_type.has_missing, dimpaths.DimPath(*rest_dims), last_type.subtypes)
        else:
            ntype = stype

        self.unpack_dims = dimpaths.DimPath(*unpack_dims)
        UnaryOpSlice.__init__(self, slice, rtype=ntype, dims=slice.dims + self.unpack_dims)#}}}

class InsertDimSlice(UnaryOpSlice):#{{{
    __slots__ = ["matchpoint","newdim"]
    def __init__(self,slice,matchpoint,ndim):
        assert len(slice.dims) >= matchpoint, "Matchpoint for dim insertion outside dimpath"
        assert ndim.shape == 1, "Length of inserted dim should be equal to 1"
        #FIXME: update va of type 
        ndims,ntype = slice.dims.insertDim(matchpoint,ndim, slice.type)
        self.matchpoint = matchpoint
        self.newdim = ndim
        UnaryOpSlice.__init__(self,slice,rtype=ntype,dims=ndims)        #}}}

class EnsureCommonDimSlice(UnaryOpSlice):#{{{
    __slots__ = ["refslices","checkpos"]
    def __init__(self,slice,refslices,checkpos,bcdim):
        self.checkpos = checkpos
        self.refslices = refslices
        ndims,ntype = slice.dims.updateDim(checkpos, bcdim, slice.type)
        UnaryOpSlice.__init__(self, slice, rtype=ntype,dims=ndims)#}}}

class BroadcastSlice(UnaryOpSlice):#{{{
    __slots__ = ["refsliceslist","plan","bcdims"]
    
    def __init__(self,slice,refslices,plan,bcdims):
        self.refsliceslist = refslices
        self.plan = plan
        self.bcdims = bcdims

        ndims = slice.dims
        ntype = slice.type
        bcpos = 0
        for pos, planelem in enumerate(plan):
            if(planelem == BCEXIST):
                assert refslices[bcpos],"No refslices for broadcast of dim: " + str(bcdims[pos])
                bcpos += 1
                ndims, ntype = ndims.updateDim(pos, bcdims[pos], ntype)
            elif(planelem == BCCOPY):
                pass
            else:
                raise RuntimeError, "Unknown plan element in plan: " + str(plan)
            
        UnaryOpSlice.__init__(self, slice, dims=ndims,rtype=ntype)#}}}

def broadcast(slices,mode="pos"):#{{{
    slicedimpaths = [s.dims for s in slices]
    if(mode == "dim"):
        bcdims, bcplan = dimpaths.planBroadcastMatchDim(slicedimpaths)
    elif(mode == "pos"):
        bcdims, bcplan = dimpaths.planBroadcastMatchPos(slicedimpaths)
    else:
        raise RuntimeError, "Unknown broadcast mode: " + str(mode)

    references = defaultdict(list)
    for bcdim in bcdims:
        for slice in slices:
            if bcdim in slice.dims:
                references[bcdim].append(slice)

    nslices = []
    for plan,slice in zip(bcplan,slices):
        nplan = []
        active_bcdims = []
        for dimpos, bcdim, planelem in zip(range(len(bcdims)),bcdims,plan):
            if(planelem == BCNEW):
                ndim = dimensions.Dim(1)
                slice = InsertDimSlice(slice,dimpos,ndim)
                active_bcdims.append(bcdim)
                nplan.append(BCEXIST)
            elif(planelem == BCENSURE):
                slice = EnsureCommonDimSlice(slice,references[bcdim],dimpos,bcdim)
                nplan.append(BCCOPY)
            elif(planelem == BCEXIST):
                active_bcdims.append(bcdim)
                nplan.append(planelem)
            else:
                nplan.append(planelem)
        if(active_bcdims):                
            slice = BroadcastSlice(slice,[references[bcdim] for bcdim in active_bcdims],nplan,bcdims)
        nslices.append(slice)
    return (nslices, bcplan)
    #}}}

def apply_broadcast_plan(slices,bcdims, bcplan):#{{{
    references = defaultdict(list)
    for bcdim in bcdims:
        for slice in slices:
            if bcdim in slice.dims:
                references[bcdim].append(slice)
   
    nslices = []
    for plan,slice in zip(bcplan,slices):
        nplan = []
        active_bcdims = []
        for dimpos, bcdim, planelem in zip(range(len(bcdims)),bcdims,plan):
            if(planelem == BCNEW):
                ndim = dimensions.Dim(1)
                slice = InsertDimSlice(slice,dimpos,ndim)
                active_bcdims.append(bcdim)
                nplan.append(BCEXIST)
            elif(planelem == BCENSURE):
                slice = EnsureCommonDimSlice(slice,references[bcdim],dimpos,bcdim)
                nplan.append(BCCOPY)
            elif(planelem == BCEXIST):
                active_bcdims.append(bcdim)
                nplan.append(planelem)
            else:
                nplan.append(planelem)
        if(active_bcdims):                
            slice = BroadcastSlice(slice,[references[bcdim] for bcdim in active_bcdims],nplan,bcdims)
        nslices.append(slice)#}}}

class PermuteDimsSlice(UnaryOpSlice):
    __slots__ = ["permute_idxs"]
    
    def __init__(self,slice,permute_idxs):
        assert len(permute_idxs) == len(slice.dims), "Number of permute indexes not equal to number of dimensions"
        
        ndims,ntype = slice.dims.permuteDims(permute_idxs,slice.type)
        self.permute_idxs = permute_idxs
        UnaryOpSlice.__init__(self,slice, dims=ndims,rtype=ntype)

class SplitDimSlice(UnaryOpSlice):
    __slots__ = ["pos","lshape","rshape"]

    def __init__(self,slice, pos,lshape,rshape,ldim,rdim):
        npath,ntype = slice.dims.updateDim(pos, (ldim,rdim),slice.type)
        self.pos = pos
        self.lshape = lshape
        self.rshape = rshape

        UnaryOpSlice.__init__(self,slice,dims=npath,rtype=ntype)

class ShapeSlice(UnaryOpSlice):
    __slots__ = ["pos"]
    def __init__(self,slice,pos):
        d = slice.dims[pos]
        ntype = rtypes.unknown
        self.pos = pos
        UnaryOpSlice.__init__(self,slice, name=d.name,rtype=ntype,dims=dimpaths.DimPath())
        

class FilterSlice(UnaryOpSlice):#{{{
    __slots__ = ["constraint"]

    def __init__(self,slice,constraint, ndim):
        stype = slice.type
        assert isinstance(stype, rtypes.TypeArray), "Filter on non-array type not possible"

        sdims = stype.dims
        if(ndim is None):
            sdims, subtype = sdims.removeDim(0, constraint, stype.subtypes[0])
        else:
            sdims, subtype = sdims.updateDim(0, ndim, stype.subtypes[0])

        if(sdims):
            ntype = rtypes.TypeArray(stype.has_missing, sdims, (subtype,))
        else:
            ntype = subtype
        
        self.constraint = constraint
        UnaryOpSlice.__init__(self, slice, rtype=ntype)#}}}

def filter(slice,constraint,seldimpath, ndim, mode="dim"):#{{{
    used_dims = [False,] * len(slice.dims)
    while True:
        #determine filter dim
        lastpos = slice.dims.matchDimPath(seldimpath)
        for filterpos in lastpos[::-1]:
             if(not used_dims[filterpos]):
                break
        else:
            break

        #pack up to and including filter dim
        packdepth = len(slice.dims) - filterpos
        if(packdepth):
            slice = PackArraySlice(slice,packdepth)
        #prepare adaptation of ndim.dependent
        if(not ndim is None):
            dep = list(ndim.dependent)
            while(len(dep) < len(constraint.dims)):
                dep.insert(0,False)

        #broadcast to constraint
        (slice,constraint),(splan,cplan) = broadcast([slice,constraint],mode=mode)

        #adapt ndim to braodcast, apply filter
        if(not ndim is None):
            if(isinstance(constraint.type,rtypes.TypeSlice)):
                ndep = dimpaths.applyPlan(dep,cplan,newvalue=True, copyvalue=True, ensurevalue=True)
            else:
                ndep = dimpaths.applyPlan(dep,cplan,newvalue=False)
            xndim = ndim.changeDependent(tuple(ndep), slice.dims)
        else:
            xndim = ndim
        slice = FilterSlice(slice,constraint,xndim)

        #adapt used_dims
        used_dims = dimpaths.applyPlan(used_dims,splan,newvalue=True,copyvalue=True,ensurevalue=True)
        
        #handle dim removal, filter_dim is used
        if(ndim is None): #dimension removed/collapsed
            packdepth -= 1
            del used_dims[len(cplan)]
        else:
            used_dims[len(cplan)] = True

        #unpack dims
        if(packdepth):
            slice = UnpackArraySlice(slice,packdepth)
            
    return slice#}}}

class FlatAllSlice(UnaryOpSlice):#{{{
    __slots__ = []

    def __init__(self,slice, ndim):
        stype = slice.type
        sdims = slice.dims
        for i in xrange(len(sdims)):
            stype = stype._removeDepDim((i+1), None)
        stype = stype._insertDepDim(1,ndim)
        dims = dimpaths.DimPath(ndim)

        UnaryOpSlice.__init__(self, slice, rtype=stype,dims=dims)#}}}

class FlatDimSlice(UnaryOpSlice):#{{{
    __slots__ = ["flatpos"]

    def __init__(self,slice,flatpos, ndim):
        stype = slice.type
        sdims = slice.dims
        sdims,stype = sdims.removeDim(flatpos,None,stype)
        sdims,stype = sdims.updateDim(flatpos-1,ndim,stype)

        self.flatpos = flatpos
        UnaryOpSlice.__init__(self, slice, rtype=stype,dims=sdims)#}}}



class UnpackTupleSlice(UnaryOpSlice):#{{{
    """A slice which is the result of unpacking a tuple slice."""
    __slots__ = ["tuple_idx"]
    
    def __init__(self, slice, idx):
        """Creates a new slice, using source `slice`, by 
        extracting the `idx` subtype.
        
        :param slice: new slice
        :param idx:   index of tuple attribute to be unpacked"""
        
        stype = slice.type
        assert isinstance(stype, rtypes.TypeTuple), "Cannot unpack slice " + \
                                str(slice.name) + " as it is not a tuple"
        
        assert 0 <= idx < len(slice.type.subtypes), \
            "Tuple index invalid, outside range of available attributes"

        ntype = slice.type.subtypes[idx]
        
        if(slice.type.fieldnames):
            name = slice.type.fieldnames[idx]
        else:
            name = "f" + str(idx)

        UnaryOpSlice.__init__(self, slice, name=name, rtype=ntype)
        self.tuple_idx = idx#}}}

class PackTupleSlice(MultiOpSlice):#{{{
    __slots__ = ["to_python"]

    def __init__(self, slices, field="data", to_python=False):
        cdim = set([slice.dims for slice in slices])
        assert len(cdim) == 1, "Packing tuple on slices with different dims"
        
        self.to_python=to_python

        fieldnames = [slice.name for slice in slices]
        subtypes = [slice.type for slice in slices]
        ntype = rtypes.TypeTuple(False, tuple(subtypes), tuple(fieldnames))
        nbookmarks = reduce(set.union,[slice.bookmarks for slice in slices])
        MultiOpSlice.__init__(self, slices, name=field, rtype=ntype, dims=iter(cdim).next(),bookmarks=nbookmarks)#}}}

class GroupIndexSlice(MultiOpSlice):
    def __init__(self, slices):
        assert len(set([slice.dims for slice in slices])) == 1, "Group index slices should have same dimension"
        assert all([isinstance(slice.type,rtypes.TypeArray) for slice in slices]), "Group index slices should be arrays"
        ndims = slices[0].dims
        
        newdims = []
        dep = (True,) * len(ndims)
        for pos, slice in enumerate(slices):
            xdep = (False,) * pos + dep
            newdims.append(dimensions.Dim(UNDEFINED,dependent=xdep,name="g" + slice.name))

        sdim = dimensions.Dim(UNDEFINED,(True,) * (len(ndims) + len(newdims)),name="g" + slices[0].type.dims[0].name)
        stype = rtypes.TypeInt64()
        rtype = rtypes.TypeArray(subtypes=(stype,), dims= dimpaths.DimPath(sdim))
        rtype = rtypes.TypeArray(subtypes=(rtype,), dims= dimpaths.DimPath(*newdims))
        MultiOpSlice.__init__(self,slices, name="groupindex", rtype=rtype, dims=ndims)


class HArraySlice(MultiOpSlice):#{{{
    __slots__ = []

    def __init__(self, slices, field="data"):
        cdim = set([slice.dims for slice in slices])
        assert len(cdim) == 1, "Packing tuple on slices with different dims"
        
        subtypes = [slice.type for slice in slices]
        assert len(set(subtypes)) == 1, "HArray can only be applied if types are equal"

        ndim = dimensions.Dim(len(slices))
        ntype = rtypes.TypeArray(False, dimpaths.DimPath(ndim), (subtypes[0],))

        nbookmarks = reduce(set.union,[slice.bookmarks for slice in slices])
        MultiOpSlice.__init__(self, slices, name=field, rtype=ntype, dims=iter(cdim).next(),bookmarks=nbookmarks)#}}}

class PackArraySlice(UnaryOpSlice):#{{{
    __slots__ = []

    def __init__(self, pslice, ndim=1):
        assert len(pslice.dims) >= ndim, "Slice does not have enough dimensions to pack as " + str(ndim) + "-dimensional array"
        
        dims = pslice.dims[-ndim:]
        has_missing = any([dim.has_missing for dim in dims])
        ntype = rtypes.TypeArray(has_missing, dims, (pslice.type,))
        UnaryOpSlice.__init__(self, pslice, rtype=ntype, dims=pslice.dims[:-ndim])#}}}

class PackListSlice(PackArraySlice):#{{{
    __slots__ = []

    def __init__(self, pslice, ndim=1):
        assert ndim == 1, "Python lists do not support multi-dimensional data"
        PackArraySlice.__init__(self, pslice, ndim)#}}}

class ConvertSlice(UnaryOpSlice):#{{{
    __slots__ = ["convertor"]

    def __init__(self, slice):
        if(slice.type == rtypes.unknown):
            ntype = rtypes.TypeAny(True)
        else:
            assert slice.type._needConversion(), "Slice does not need conversion"
            ntype = slice.type._setNeedConversion(False)

        self.convertor = slice.type._getConvertor()
        UnaryOpSlice.__init__(self, slice, rtype=ntype)#}}}

class FreezeSlice(UnaryOpSlice):#{{{
    __slots__ = []
#}}}

def ensure_frozen(slice):#{{{
    if(freeze_protocol.needFreeze(slice.type)):
        return FreezeSlice(slice)
    else:
        return slice#}}}

def ensure_converted(slice):#{{{
    if(slice.type._needConversion()):
        return ConvertSlice(slice)
    else:
        return slice#}}}

class FuncSlice(UnaryOpSlice):#{{{
    __slots__ = ["exec_func", "type_func", "params", "kwds"]
    
    def __init__(self,slice, exec_func, type_func, ndims, ntype,  *params, **kwds):
        UnaryOpSlice.__init__(self,slice, rtype=ntype, dims=ndims)
        self.exec_func = exec_func
        self.type_func = type_func
        self.params = params
        self.kwds = kwds#}}}
       
class MapSeqSlice(FuncSlice):#{{{
    __slots__ = []
    def __init__(self, slice, exec_func, type_func, *params, **kwds): 
        """Creates a new slice, and sets attributes.

        :param slice: Source slice func is applied on.
        :param exec_func: function to be applied
        :param type_func: function to determine outtype."""
        assert slice.dims, str(exec_func) + " can only be applied on slice " + \
                                    "with at least one dimension"
        
        ntype = type_func(slice.type, exec_func)
        FuncSlice.__init__(self,slice,dim,exec_func, type_func, slice.dims, ntype, *params, **kwds)#}}}

class MapSlice(FuncSlice):#{{{
    __slots__ = []
    def __init__(self, slice, exec_func, type_func, *params, **kwds): 
        """Creates a new slice, and sets attributes.

        :param slice: Source slice func is applied on.
        :param exec_func: function to be applied
        :param type_func: function to determine outtype."""
        ntype = type_func(slice.type, exec_func)
        FuncSlice.__init__(self,slice,dim,exec_func, type_func, slice.dims, ntype, *params, **kwds)#}}}

class AggregrateSlice(FuncSlice):#{{{
    __slots__ = []
    def __init__(self, slice, exec_func, type_func, *params, **kwds): 
        """Creates a new slice, and sets attributes.

        :param slice: Source slice func is applied on.
        :param exec_func: function to be applied
        :param type_func: function to determine outtype."""
        ndim = kwds.pop("ndim",1)

        assert len(slice.dims) >= ndim, "Slice does not have enough dimensions for aggregration"
        ndims = slice.dims[:-ndim]
        ntype = type_func(slice.type, slice.dims[-ndim:], exec_func)
        FuncSlice.__init__(self, slice,dim,exec_func, type_func, ndims, ntype, *params, **kwds)#}}}

class BinFuncOpSlice(MultiOpSlice):#{{{
    __slots__ = ["funcname","sig"]
    def __init__(self, lslice, rslice, funcname, sig, outparam, dims=None):
        self.funcname = funcname
        self.sig = sig

        if(dims is None):
            assert lslice.dims == rslice.dims, "Slice dims not equal, and no dim given"
            dims = lslice.dims
        nbookmarks = lslice.bookmarks | rslice.bookmarks

        MultiOpSlice.__init__(self, (lslice, rslice), name=outparam.name, rtype=outparam.type, dims=dims, bookmarks=nbookmarks)#}}}

class BinFuncElemOpSlice(BinFuncOpSlice):#{{{
    __slots__ = ["allow_partial_bc"]

    def __init__(self, funcname, sig, outparam, left, right, allow_partial_bc=False):
        dims = left.dims
        self.allow_partial_bc = allow_partial_bc
        assert all([d1 == d2 for d1, d2 in zip(dims, right.dims)]), \
                    "Dimensions of slices do not match"
        
        BinFuncOpSlice.__init__(self, left, right, funcname, sig, outparam, dims)#}}}

class UnaryFuncOpSlice(UnaryOpSlice):#{{{
    __slots__ = ["funcname","sig", "kwargs"]
    def __init__(self, slice, funcname, sig, outparam, dims=None, **kwargs):
        self.funcname = funcname
        self.sig = sig
        self.kwargs = kwargs

        if(dims is None):
            dims = slice.dims
        UnaryOpSlice.__init__(self, slice, name=outparam.name, rtype=outparam.type, dims=dims)#}}}

class UnaryFuncElemOpSlice(UnaryFuncOpSlice):#{{{
    __slots__ = []
    
    def __init__(self, funcname, sig, outparam, slice, **kwargs):
        UnaryFuncOpSlice.__init__(self, slice, funcname, sig, outparam, **kwargs)
        #}}}

class UnaryFuncSeqOpSlice(UnaryFuncOpSlice):#{{{
    __slots__ = ["packdepth"]

    
    def __init__(self, funcname, sig, outparam, packdepth, slice, **kwargs):
        self.packdepth = packdepth
        UnaryFuncOpSlice.__init__(self, slice, funcname, sig, outparam, **kwargs)
        #}}}


class UnaryFuncAggregateOpSlice(UnaryFuncOpSlice):#{{{
    __slots__ = ["packdepth"]
    
    def __init__(self, funcname, sig, outparam, packdepth, slice, **kwargs):
        self.packdepth = packdepth
        sdims, ntype = slice.dims.removeDim(len(slice.dims) - packdepth, (funcname,outparam), outparam.type)

        UnaryFuncOpSlice.__init__(self, slice, funcname, sig, outparam, dims=sdims, **kwargs)
        self.type = ntype
        #}}}

