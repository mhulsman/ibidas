import copy

from constants import *
from utils import util

_delay_import_(globals(),"itypes","dimensions","rtypes")
_delay_import_(globals(),"itypes.type_attribute_freeze","freeze_protocol")
_delay_import_(globals(),"typeops")


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

    def modify(self, new_dims=None, realias=False, no_copy = False):
        """Returns (modified) copy.

        Parameters
        ----------
        new_dims : optional. as dict {olddim: newdim}
                             or as tuple (newdim1, newdim2)
        realias :  optional. replaces id with new id, and copies
                             old id to last_id
        """
        copied = False
        res = self
        if(new_dims):
            ndims = list(res.dims)
            changed=False
            for dimid_old, dim_new in new_dims.iteritems():
                cres = util.contained_in(dimid_old, ndims)
                if(cres is False):
                    continue
                changed=True
                
                if(dim_new is None):
                    del ndims[cres[1]-1]
                elif(isinstance(dim_new, tuple)):
                    ndims = ndims[:(cres[1]-1)] + list(dim_new) + ndims[cres[1]:]
                else:
                    ndims[cres[1]-1] = dim_new

            if(changed is True):
                copied = True
                if(not no_copy):
                    res = copy.copy(self)
                res.dims = tuple(ndims)
        if(realias):
            if(not no_copy and not copied):
                res = copy.copy(self)
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
    __slots__ = ['source_ids']
    def __init__(self, source_slices, name, rtype=rtypes.unknown, 
                            dims=(), sid=None, last_id=None):
        self.source_ids = tuple([slice.id for slice in source_slices])
        Slice.__init__(self, name, rtype, dims, sid, last_id)

class UnpackArraySlice(OpSlice):
    """An slice which is the result of unpacking a source slice."""
    __slots__ = []
    
def sunpack_array(slice):
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
    
    self = UnpackArraySlice((slice,), slice.name, 
                                    stype.subtypes[0], slice.dims + ndims)
    return self

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
        stype = slice.type
        assert isinstance(stype, rtypes.TypeTuple), "Cannot unpack slice " + \
                                str(slice.name) + " as it is not a tuple"
        
        assert 0 <= idx < len(slice.type.subtypes), \
            "Tuple index invalid, outside range of available attributes"
        ntype = freeze_protocol.freeze(slice.type.subtypes[idx])
        
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
        ntype = rtypes.TypeTuple(False, tuple(subtypes), tuple(fieldnames))
        
        OpSlice.__init__(self, slices, field, ntype, iter(cdim).next())

class PackArraySlice(OpSlice):
    __slots__ = []

    def __init__(self, name, pslice, dim):
        assert pslice.dims, "Cannot pack as array a slice without dimension"
        dim = pslice.dims[-1]
        in_type = pslice.type
        ntype = rtypes.TypeArray(dim.has_missing, (dim, ), (in_type,), need_freeze=False)
        OpSlice.__init__(self, (pslice,), pslice.name, ntype, pslice.dims[:-1])

class PackListSlice(OpSlice):
    __slots__ = []

    def __init__(self, name, pslice, dim):
        assert pslice.dims, "Cannot pack as python list a slice without dimension"
        dim = pslice.dims[-1]
        in_type = pslice.type
        ntype = rtypes.TypeArray(dim.has_missing, (dim, ), (in_type,), need_freeze=True)
        OpSlice.__init__(self, (pslice,), pslice.name, ntype, pslice.dims[:-1])

class MapSeqSlice(OpSlice):
    """An slice which is the result of unpacking a source slice."""
    __slots__ = ["exec_func", "type_func", "params", "kwds"]
    def __init__(self, name, slice, dim, exec_func, type_func, *params, **kwds): 
        """Creates a new slice, and sets attributes.

        Parameters
        ----------
        slice: Source slice func is applied on.
        exec_func: function to be applied
        type_func: function to determine outtype."""
        assert slice.dims, str(exec_func) + " can only be applied on slice " + \
                                    "with at least one dimension"
        ntype = type_func(slice.type, exec_func)
        
        OpSlice.__init__(self,(slice,), slice.name, 
                                        ntype, slice.dims)
        self.exec_func = exec_func
        self.type_func = type_func
        self.params = params
        self.kwds = kwds

class MapSlice(OpSlice):
    """An slice which is the result of unpacking a source slice."""
    __slots__ = ["exec_func", "type_func"]
    def __init__(self, name, slice, dim, exec_func, type_func): 
        """Creates a new slice, and sets attributes.

        Parameters
        ----------
        slice: Source slice func is applied on.
        exec_func: function to be applied
        type_func: function to determine outtype."""
        ntype = type_func(slice.type, exec_func)
        
        OpSlice.__init__(self,(slice,), slice.name, 
                                        ntype, slice.dims)
        self.exec_func = exec_func
        self.type_func = type_func

class AggregateSlice(OpSlice):
    """An slice which is the result of unpacking a source slice."""
    __slots__ = ["exec_func", "type_func"]
    
    def __init__(self, name, slice, dim, exec_func, type_func):
        """Creates a new slice, and sets attributes.

        Parameters
        ----------
        slice: Source slice func is applied on.
        exec_func: function to be applied
        type_func: function to determine outtype."""

        assert slice.dims, "Aggregration can only be applied on slice " + \
                                    "with at least one dimension"
        ntype = type_func(slice.type, slice.dims[-1], exec_func)
        ndims = slice.dims[:-1]
        
        OpSlice.__init__(self, (slice,), slice.name, 
                                        ntype, ndims)
        self.exec_func = exec_func
        self.type_func = type_func

class FreezeSlice(OpSlice):
    """An slice which is the result of unpacking a source slice."""
    __slots__ = []
    
def freeze_slice(name, slice, dim):
    """Creates a new slice, and sets attributes.

    Parameters
    ----------
    slice: Source slice to be freezed."""

    if(not freeze_protocol.requireFreeze(slice.type)):
        return slice
    
    ntype = freeze_protocol.freeze(slice.type)
    return FreezeSlice((slice,), slice.name, 
                                    ntype, slice.dims)

class BinOpSlice(OpSlice):
    __slots__ = ['source1_id', 'source2_id']
    def __init__(self, slice1, slice2, rtype=rtypes.unknown, dims=(), 
                                    sid=None, last_id=None, outidx=None):
        if(slice1.name == slice2.name):
            nname = slice1.name
        else:
            nname = "result"
            if(outidx):
                nname += str(outidx)
        
        OpSlice.__init__(self, (slice1, slice2), nname, rtype, 
                         dims, sid, last_id)

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

