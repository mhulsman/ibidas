import copy

from constants import *
from utils import util
from itypes import rtypes

_delay_import_(globals(),"itypes","dimensions","typeops")
_delay_import_(globals(),"itypes.type_attribute_freeze","freeze_protocol")


#pylint: disable-msg=E1101
sliceid = util.seqgen().next
class Slice(object):
    """A slice represents a attribute and set of dimensions in the data.
    Each slice has an id for unique identification, an attribute name
    for accessing it, a type describing the contents of the slice,
    a dims attribute describing the packaging of the attribute, 
    and a last_id, describing a slice from which the current
    slice has been copied (id's within a table should be unique).
    
    Note: one can have multiple slices with dissimilar dimensions
    but the same id. An id represents similarity of the content 
    on element level, dims the packaging."""

    __slots__ = ['id', 'name', 'type', 'dims', 'last_id']


    def __init__(self, name, rtype = rtypes.unknown, dims=(), 
                                                sid=None, last_id=None):
        """Creates a slice object.

        Parameters
        ----------
        name: name of slice (string)
        type: type of slice, optional, default = unknown
        dims: tuple of Dim objects, optional, default = ()
        id:   id of slice, optional, default = auto-generated
        last_id: if copied with new `id`, this should refer to `id`
                   of copied object
        """
        if(sid is None):
            self.id = sliceid()
        else:
            self.id = sid
        assert isinstance(name, str), "Name of slice should be a string"
        assert (name.lower() == name), "Name should be in lowercase"
        self.name = name

        assert isinstance(rtype, rtypes.TypeUnknown), "Invalid type given"
        self.type = rtype

        assert isinstance(dims, tuple), \
                                    "Dimensions of a slice should be a tuple"
        self.dims = dims
        self.last_id = last_id
    
    def copy(self, realias=False):
        res =  copy.copy(self)
        if(realias):
            res.last_id = res.id
            res.id = sliceid()
        return res

    def setName(self, name):
        assert isinstance(name, str), "Name of slice should be a string"
        assert (name.lower() == name), "Name should be in lowercase"
        self.name = name

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.id == other.id and self.dims == other.dims)

    def __hash__(self):
        return hash(self.id) ^ hash(self.dims)

    def __repr__(self, last_dims=()):
        res = self.name
        res += "="
        if(self.dims):
            dimstr = []
            for pos, dim in enumerate(self.dims):
                if(len(last_dims) > pos and dim == last_dims[pos]):
                    dimstr.append(".")
                else:
                    dimstr.append("[" + str(dim) + "]")
            res += "<".join(dimstr) + "<"
                
        res += str(self.type)
        return res

class OpSlice(Slice):
    __slots__ = ['source']
    def __init__(self, source_slices, name, rtype=rtypes.unknown, 
                            dims=(), sid=None, last_id=None):
        self.source = tuple([slice.id for slice in source_slices])
        Slice.__init__(self, name, rtype, dims, sid, last_id)

class UnpackArraySlice(OpSlice):
    """An slice which is the result of unpacking a source slice."""
    __slots__ = []

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
            rest_dims = tuple()
        else:
            unpack_dims = ndims[:ndim]
            rest_dims = ndims[ndim:]

        if(rest_dims):
            ntype = rtypes.TypeArray(stype.has_missing, rest_dims, stype.subtypes, data_state=DATA_NORMAL)
            OpSlice.__init__(self, (slice,), slice.name, 
                                            ntype, slice.dims + unpack_dims)
        else:
            OpSlice.__init__(self, (slice,), slice.name, 
                                            stype.subtypes[0], slice.dims + unpack_dims)
        

class UnpackTupleSlice(OpSlice):#{{{
    """A slice which is the result of unpacking a tuple slice."""
    __slots__ = ["slice_idx"]
    
    def __init__(self, slice, idx):
        """Creates a new slice, using source `slice`, by 
        extracting the `idx` subtype.
        
        Parameters
        ----------
        slice: new slice
        idx:   index of tuple attribute to be unpacked"""
        slice = ensure_normal_or_frozen(slice)
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

        OpSlice.__init__(self, (slice,), name, 
                                        ntype, slice.dims)
        self.slice_idx = idx#}}}

class PackTupleSlice(OpSlice):
    __slots__ = ["to_python"]

    def __init__(self, slices, field="data", to_python=False):
        cdim = set([slice.dims for slice in slices])
        
        assert len(cdim) == 1, "Packing tuple on slices with different dims"
        
        self.to_python=to_python
        fieldnames = [slice.name for slice in slices]
        subtypes = [slice.type for slice in slices]
        ntype = rtypes.TypeTuple(False, tuple(subtypes), tuple(fieldnames),data_state=DATA_NORMAL)
        
        OpSlice.__init__(self, slices, field, ntype, iter(cdim).next())

class PackArraySlice(OpSlice):
    __slots__ = []

    def __init__(self, pslice, ndim=1):
        assert pslice.dims, "Cannot pack as array a slice without dimension"
        
        dims = pslice.dims[-ndim:]
        in_type = pslice.type
        has_missing = any([dim.has_missing for dim in dims])
        ntype = rtypes.TypeArray(has_missing, dims, (in_type,), data_state=DATA_NORMAL)

        OpSlice.__init__(self, (pslice,), pslice.name, ntype, pslice.dims[:-ndim])


class PackListSlice(PackArraySlice):
    __slots__ = []

    def __init__(self, pslice, ndim=1):
        assert ndim == 1, "Python lists do not support multi-dimensional data"
        PackArraySlice.__init__(self,pslice,dim,ndim)

class RedimSlice(OpSlice):
    __slots__ = []
    def __init__(self, slice, ndims,  *params, **kwds):
        OpSlice.__init__(self,(slice,), slice.name, 
                                        slice.type, ndims)

class ConvertSlice(OpSlice):
    __slots__ = ["convertor"]

    def __init__(self, pslice):
        if(slice.type == rtypes.unknown):
            ntype = rtypes.TypeAny(True,data_state=DATA_NORMAL)
        else:
            assert slice.type.data_state == DATA_INPUT, "Slice has not DATA_INPUT as state"
            ntype = slice.type.copy(data_state=DATA_NORMAL)
        self.convertor = convertors.getConvertor(slice.type)
        OpSlice.__init__(self, (pslice,), pslice.name, ntype, pslice.dims)


class FreezeSlice(OpSlice):
    __slots__ = ["convertor"]

    def __init__(self, pslice):
        assert slice.type.data_state == DATA_NORMAL, "Slice does not need to frozen"
        ntype = freeze_protocol.freeze(slice.type)
        OpSlice.__init__(self, (pslice,), pslice.name, ntype, pslice.dims)

class UnFreezeSlice(OpSlice):
    __slots__ = ["convertor"]

    def __init__(self, pslice):
        assert freeze_protocol.need_unfreeze(slice.type), "Slice does not need to be unfrozen"
        ntype = freeze_protocol.unfreeze(slice.type)
        OpSlice.__init__(self, (pslice,), pslice.name, ntype, pslice.dims)

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

class FuncSlice(OpSlice):
    __slots__ = ["exec_func", "type_func", "params", "kwds"]
    def __init__(self,slice, exec_func, type_func, ndims, ntype,  *params, **kwds):
        assert slice.type.data_state == DATA_NORMAL, "Data should be in normal state"        
        OpSlice.__init__(self,(slice,), slice.name, ntype, ndims)
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

class BinOpSlice(OpSlice):
    __slots__ = []
    def __init__(self, slice1, slice2, rtype=rtypes.unknown, dims=(), 
                                    sid=None, last_id=None, outidx=None):
        assert slice1.type.data_state == DATA_NORMAL, "Data should be in normal state"        
        assert slice2.type.data_state == DATA_NORMAL, "Data should be in normal state"        

        if(slice1.name == slice2.name):
            nname = slice1.name
        else:
            nname = "result"
            if(outidx):
                nname += str(outidx)
        
        OpSlice.__init__(self, (slice1, slice2), nname, rtype, dims, sid, last_id)

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

class UnaryElemOpSlice(OpSlice):
    """An slice which is the result of unpacking a source slice."""
    __slots__ = ["oper", "op"]
    
    def __init__(self, slice, op, outtype=None):
        """Creates a new slice, and sets attributes.

        Parameters
        ----------
        slice: Source slice func is applied on.
        op: operator (python name)
        outtype: ibidas out type or None."""
        ntype, oper = typeops.unop_type(slice.type, op, outtype)

        OpSlice.__init__(self, (slice,), slice.name, ntype, slice.dims)
        self.oper = oper
        self.op = op

