import copy

from constants import *
from utils import util
from itypes import rtypes,dimpaths
from query_graph import Node


_delay_import_(globals(),"itypes","dimensions","typeops","convertors","casts")
_delay_import_(globals(),"itypes.type_attribute_freeze","freeze_protocol")

#pylint: disable-msg=E1101
sliceid = util.seqgen().next
class Slice(Node):
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

        Parameters
        ----------
        name: name of slice (string)
        type: type of slice, optional, default = unknown
        dims: tuple of Dim objects, optional, default = ()
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
        return res

class UnaryOpSlice(Slice):
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
        Slice.__init__(self, name, rtype, dims, bookmarks)

class MultiOpSlice(Slice):
    __slots__ = ['sources']
    def __init__(self, source_slices, name, rtype=rtypes.unknown, 
                            dims=dimpaths.DimPath(), bookmarks=set()):
        self.sources = tuple(source_slices)
        Slice.__init__(self, name, rtype, dims, bookmarks)

class LinkSlice(UnaryOpSlice):
    __slots__ = ['link']
    def __init__(self, source, link, name, rtype=rtypes.unknown, dims=dimpaths.DimPath(), bookmarks=set()):
        assert isinstance(link,representor.Representor),"Link of LinkSlice should be a representor"
        UnaryOpSlice.__init__(self, source, name, rtype, dims, bookmarks)

class DataSlice(Slice):
    __slots__ = ['data']
    def __init__(self, data, name=None, rtype=None, dims=dimpaths.DimPath(), bookmarks=set()):
        self.data = data
        Slice.__init__(self, name, rtype, dims, bookmarks)


class ChangeBookmarkSlice(UnaryOpSlice):
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
        
        UnaryOpSlice.__init__(self, source, bookmarks=nbookmarks)

class ChangeNameSlice(UnaryOpSlice):
    __slots__ = []
    def __init__(self,source, new_name):
        UnaryOpSlice.__init__(self, source, name=new_name)

class ChangeDimPathSlice(UnaryOpSlice):
    __slots__ = []
    def __init__(self,source, new_dims):
        UnaryOpSlice.__init__(self, source, dims=new_dims)

class CastSlice(UnaryOpSlice):
    __slots__ = ["cast_name"]
    def __init__(self, source, new_type):
        cast_name = casts.canCast(source.type,new_type)
        assert not cast_name is False, "Cannot cast " + str(source.type) + " to " + str(new_type)
        self.cast_name = cast_name
        UnaryOpSlice.__init__(self, source, rtype=new_type)
        

class DetectTypeSlice(UnaryOpSlice):
    __slots__ = []


class UnpackArraySlice(UnaryOpSlice):
    """An slice which is the result of unpacking a source slice."""
    __slots__ = ["unpack_dims"]

    def __init__(self,slice,ndim=None):
        """Creates a new slice, and sets attributes.

        Parameters
        ----------
        slice: Source slice to be unpacked"""
        stype = slice.type
        assert isinstance(stype, rtypes.TypeArray), "Cannot unpack slice " + \
                                str(slice.name) + " as it is not an array"
        if(not stype.dims):
            ndims = (dimensions.Dim(UNDEFINED, variable=1, 
                                    has_missing=stype.has_missing),)
        else:
            ndims = stype.dims

        if(ndim is None):
            unpack_dims = ndims
            rest_dims = dimpaths.DimPath()
        else:
            unpack_dims = ndims[:ndim]
            rest_dims = ndims[ndim:]
        self.unpack_dims = unpack_dims

        if(rest_dims):
            ntype = rtypes.TypeArray(stype.has_missing, rest_dims, stype.subtypes)
            UnaryOpSlice.__init__(self, slice, rtype=ntype, dims=slice.dims + unpack_dims)
        else:
            UnaryOpSlice.__init__(self, slice, rtype=stype.subtypes[0], dims=slice.dims + unpack_dims)
        

class InsertDimSlice(UnaryOpSlice):
    __slots__ = ["matchpoint","newdim"]
    def __init__(self,slice,matchpoint,ndim):
        assert len(slice.dims) > matchpoint, "Matchpoint for dim insertion outside dimpath"
        #FIXME: update va of type 
        ndims = slice.dims[:matchpoint] + (ndim,) + slice.dims[matchpoint:].updateDimVariable()
        ntype = slice.type.updateDimVariable(insertpoint=-(len(slice.dims) - matchpoint))

        self.matchpoint = matchpoint
        self.newdim = ndim
        UnaryOpSlice.__init__(self,slice,rtype=ntype,dims=ndims)        

class UnpackTupleSlice(UnaryOpSlice):#{{{
    """A slice which is the result of unpacking a tuple slice."""
    __slots__ = ["tuple_idx"]
    
    def __init__(self, slice, idx):
        """Creates a new slice, using source `slice`, by 
        extracting the `idx` subtype.
        
        Parameters
        ----------
        slice: new slice
        idx:   index of tuple attribute to be unpacked"""
        
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

class PackTupleSlice(MultiOpSlice):
    __slots__ = ["to_python"]

    def __init__(self, slices, field="data", to_python=False):
        cdim = set([slice.dims for slice in slices])
        assert len(cdim) == 1, "Packing tuple on slices with different dims"
        
        self.to_python=to_python

        fieldnames = [slice.name for slice in slices]
        subtypes = [slice.type for slice in slices]
        ntype = rtypes.TypeTuple(False, tuple(subtypes), tuple(fieldnames))
        nbookmarks = reduce(set.union,[slice.bookmarks for slice in slices])
        MultiOpSlice.__init__(self, slices, name=field, rtype=ntype, dims=iter(cdim).next(),bookmarks=nbookmarks)

class HArraySlice(MultiOpSlice):
    __slots__ = []

    def __init__(self, slices, field="data"):
        cdim = set([slice.dims for slice in slices])
        assert len(cdim) == 1, "Packing tuple on slices with different dims"
        
        subtypes = [slice.type for slice in slices]
        assert len(set(subtypes)) == 1, "HArray can only be applied if types are equal"

        ndim = dimensions.Dim(len(slices))
        ntype = rtypes.TypeArray(False, dimpaths.DimPath(ndim), (subtypes[0],))

        nbookmarks = reduce(set.union,[slice.bookmarks for slice in slices])
        MultiOpSlice.__init__(self, slices, name=field, rtype=ntype, dims=iter(cdim).next(),bookmarks=nbookmarks)

class PackArraySlice(UnaryOpSlice):
    __slots__ = []

    def __init__(self, pslice, ndim=1):
        assert len(pslice.dims) >= ndim, "Slice does not have enough dimensions to pack as " + str(ndim) + "-dimensional array"
        
        dims = pslice.dims[-ndim:]
        has_missing = any([dim.has_missing for dim in dims])
        ntype = rtypes.TypeArray(has_missing, dims, (pslice.type,))
        UnaryOpSlice.__init__(self, pslice, rtype=ntype, dims=pslice.dims[:-ndim])


class PackListSlice(PackArraySlice):
    __slots__ = []

    def __init__(self, pslice, ndim=1):
        assert ndim == 1, "Python lists do not support multi-dimensional data"
        PackArraySlice.__init__(self, pslice, ndim)


class ConvertSlice(UnaryOpSlice):
    __slots__ = ["convertor"]

    def __init__(self, slice):
        if(slice.type == rtypes.unknown):
            ntype = rtypes.TypeAny(True)
        else:
            assert slice.type.data_state == DATA_INPUT, "Slice has not DATA_INPUT as state"
            ntype = slice.type.copy()
            ntype.data_state = DATA_NORMAL
            ntype.attr.pop('convertor',None)

        self.convertor = convertors.getConvertor(slice.type)
        UnaryOpSlice.__init__(self, slice, rtype=ntype)


class FreezeSlice(UnaryOpSlice):
    __slots__ = []

    def __init__(self, pslice):
        assert freeze_protocol.need_freeze(pslice.type), "Slice does not need to frozen"
        ntype = freeze_protocol.freeze(slice.type)
        UnaryOpSlice.__init__(self, pslice, rtype=ntype)

class UnFreezeSlice(UnaryOpSlice):
    __slots__ = []

    def __init__(self, pslice):
        assert freeze_protocol.need_unfreeze(slice.type), "Slice does not need to be unfrozen"
        ntype = freeze_protocol.unfreeze(slice.type)
        UnaryOpSlice.__init__(self, pslice, rtype=ntype)

def ensure_frozen(slice):
    if(slice.type == rtypes.unknown or slice.type.data_state == DATA_INPUT):
        slice = ConvertSlice(slice)
    
    if(freeze_protocol.need_freeze(slice.type)):
        return FreezeSlice(slice)
    else:
        return slice

def ensure_normal(slice):
    if(slice.type == rtypes.unknown or slice.type.data_state == DATA_INPUT):
        return ConvertSlice(slice)
    elif(freeze_protocol.need_unfreeze(slice.type)):
        return UnFreezeSlice(slice)
    else:
        return slice

def ensure_normal_or_frozen(slice):
    if(slice.type == rtypes.unknown or slice.type.data_state == DATA_INPUT):
        return ConvertSlice(slice)
    else:
        return slice

class FuncSlice(UnaryOpSlice):
    __slots__ = ["exec_func", "type_func", "params", "kwds"]
    
    def __init__(self,slice, exec_func, type_func, ndims, ntype,  *params, **kwds):
        UnaryOpSlice.__init__(self,slice, rtype=ntype, dims=ndims)
        self.exec_func = exec_func
        self.type_func = type_func
        self.params = params
        self.kwds = kwds
       

class MapSeqSlice(FuncSlice):
    __slots__ = []
    def __init__(self, slice, exec_func, type_func, *params, **kwds): 
        """Creates a new slice, and sets attributes.

        Parameters
        ----------
        slice: Source slice func is applied on.
        exec_func: function to be applied
        type_func: function to determine outtype."""
        assert slice.dims, str(exec_func) + " can only be applied on slice " + \
                                    "with at least one dimension"
        
        ntype = type_func(slice.type, exec_func)
        FuncSlice.__init__(self,slice,dim,exec_func, type_func, slice.dims, ntype, *params, **kwds)

class MapSlice(FuncSlice):
    __slots__ = []
    def __init__(self, slice, exec_func, type_func, *params, **kwds): 
        """Creates a new slice, and sets attributes.

        Parameters
        ----------
        slice: Source slice func is applied on.
        exec_func: function to be applied
        type_func: function to determine outtype."""
        ntype = type_func(slice.type, exec_func)
        FuncSlice.__init__(self,slice,dim,exec_func, type_func, slice.dims, ntype, *params, **kwds)


class AggregrateSlice(FuncSlice):
    __slots__ = []
    def __init__(self, slice, exec_func, type_func, *params, **kwds): 
        """Creates a new slice, and sets attributes.

        Parameters
        ----------
        slice: Source slice func is applied on.
        exec_func: function to be applied
        type_func: function to determine outtype."""
        ndim = kwds.pop("ndim",1)

        assert len(slice.dims) >= ndim, "Slice does not have enough dimensions for aggregration"
        ndims = slice.dims[:-ndim]
        ntype = type_func(slice.type, slice.dims[-ndim:], exec_func)
        FuncSlice.__init__(self, slice,dim,exec_func, type_func, ndims, ntype, *params, **kwds)

class BinOpSlice(MultiOpSlice):
    __slots__ = []
    def __init__(self, slice1, slice2, rtype=rtypes.unknown, dims=dimpaths.DimPath(), outidx=None):
        if(slice1.name == slice2.name):
            nname = slice1.name
        else:
            nname = "result"
            if(outidx):
                nname += str(outidx)

        nbookmarks = slice1.bookmarks | slice2.bookmarks
        MultiOpSlice.__init__(self, (slice1, slice2), name=nname, rtype=rtype, dims=dims, bookmarks=nbookmarks)

class BinElemOpSlice(BinOpSlice):
    __slots__ = ['oper', 'op']

    def __init__(self, slice1, slice2, op, outtype=None, outidx=None):
        ntype, oper = typeops.binop_type(slice1.type, slice2.type, op, outtype)
        dim1 = slice1.dims
        dim2 = slice2.dims
        if(len(dim1) < len(dim2)):
            dim2, dim1 = dim1, dim2

        assert all([d1 == d2 for d1, d2 in zip(dim1, dim2)]), \
                    "Dimensions of slices do not match"
        
        BinOpSlice.__init__(self, slice1, slice2, ntype, dim1, outidx=outidx)
        self.oper = oper
        self.op = op

class UnaryElemOpSlice(UnaryOpSlice):
    __slots__ = ["oper", "op"]
    
    def __init__(self, slice, op, outtype=None):
        """Creates a new slice, and sets attributes.

        Parameters
        ----------
        slice: Source slice func is applied on.
        op: operator (python name)
        outtype: ibidas out type or None."""
        ntype, oper = typeops.unop_type(slice.type, op, outtype)
        UnaryOpSlice.__init__(self, (slice,), rtype=ntype)
        self.oper = oper
        self.op = op

