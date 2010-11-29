"""rtypes: main type hiearchy

The type system is a hierarchical system, where each child type adds some
information w.r.t to its parent type. 

Root type is TypeUnknown, with as child TypeAny. Remaining types are divided in
- TypeScalar:   all scalar types, such as integers, strings.
- TypeTuple:    values containing multiple attributes/fields/elements, each
                with a (possibly different) type. 
- TypeArray:    collection of values of similar type, in an storage structure 
                described by dimensions
- TypeFunc:     methods/functions (to do)

Adding new types
----------------
Adding a new type is as simple as inheriting from one of the types,
and calling the function addType(>typeclass<), e.g. 

TypeDNASequence(TypeString):
    name = "DNASequence"
addType(TypeDNASequence)

Extra attributes such as _dtype, _scalar can be set to adapt the 
internal numpy representation. Examples can be found in the source code of 
this file. 

When the type is added, one can add operations (type_ops.py),
casts (type_casts.py) and automatic object detection (scanners.py).


Hierarchy rules
---------------
One should only inherit if the child type can only contain a subset 
of the values of the parent type. 

Operations on the parent type should be applicable on the child type 
(but one can override them for more type-specific behaviour).

Multiple inheritance is allowed, e.g. TypeUInt8(TypeUInt16, TypeInt16)

It is currently not possible to add types within the hieararchy (e.g.
a type inbetween TypeUnknown and any of the other types), 
without modifying the source in this file or some hackish runtime class modification. 
"""


import numpy
import operator
from collections import defaultdict

from ..constants import *

_delay_import_(globals(),"dimensions")
_delay_import_(globals(),"casts")
_delay_import_(globals(),"type_attribute_freeze")


#}}}

#typename to type class dictionary
__typenames__ = {}
#children types classes for each parent type class
__typechildren__ = defaultdict(list)

from ibidas import constants

def addType(newtypecls):
    """Adds a type `newtypecls`
    to the type hiearchy"""

    __typenames__[newtypecls.name] = newtypecls
    for basecls in newtypecls.__bases__:
        __typechildren__[basecls].append(newtypecls)



#Combining types requires combining type attribures
#such as has_missing. 
#Here, you can register functions which do such combining
#for common type attributes
_type_attribute_common = {}
def addTypeAttribute(name, common_visitor):
    _type_attribute_common[name] = common_visitor




class TypeUnknown(object):#{{{
    """Type which represents that no information is known about the values.
    As this type has only one possible state, 
    a module singleton is available through rtypes.unknown"""
    
    name = "?"
    _dtype = "object"
    _scalar = numpy.object
    _defval = None
    data_state=DATA_NORMAL

    @classmethod
    def commonType(cls, type1, type2):
        """Returns a common supertype for type1 and type2"""
        return unknown

    def toNumpy(self):
        """Returns dtype of a numpy container which
        can hold this type efficiently.
        """
        return numpy.dtype(self._dtype)

    def toScalar(self):
        """Returns class of a numpy scalar which can
        hold this type efficiently"""
        return self._scalar

    def toDefval(self):
        """Returns default value."""
        return self._defval
    
    @classmethod
    def getDescendantTypes(cls):
        """Returns descendant types classes and cls
        as a list"""
        if(not hasattr(cls, "_desc_types")):
            desc_types = [tcls.getDescendantTypes() 
                                for tcls in __typechildren__[cls]]
            #flatten list (concat each list in list)
            if(desc_types):
                desc_types = sum(desc_types, [])
            desc_types.append(cls)
            cls._desc_types = desc_types
        return cls._desc_types

    def getSubType(self, subtype_id):
        """Returns subtype. For unknown type, this
        is always the unknown type"""
        return unknown

    def __eq__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        return other.__class__ is self.__class__
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__eq__(other)
    
    def __ge__(self, other):
        return True

    def __gt__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        return False

    def __hash__(self):
        return hash(self.__class__)

    def copy(self):
        """Returns copy of this type"""
        return self.__class__()
   
    def redim_var(self, deplen, var_adapt, redim_cache):
        return self

    def __repr__(self):
        return self.name
addType(TypeUnknown)#}}}

#unknown singleton
unknown = TypeUnknown()

class TypeAny(TypeUnknown):#{{{
    """Type which represents that any value is allowed"""

    name = "any"

    def __init__(self, has_missing=False, **attr):
        """
        Creates type object.

        Parameters
        ----------
        has_missing : bool, optional
        scanner : scanner  object, optional
        """
        TypeUnknown.__init__(self)
        self.has_missing = has_missing
        self.attr = attr

    def __getattr__(self,name):
        if(name in self.attr):
            return self.attr[name]
        else:
            raise AttributeError, "Cannot find " + name
    
    @classmethod
    def commonType(cls, type1, type2):
        res = cls(type1.has_missing or type2.has_missing)
        res.setCommonAttributes(type1, type2)
        return res

   
    def setCommonAttributes(self, type1, type2):
        for attrkey in set(type1.attr.keys() + type2.attr.keys()):
            if not attrkey in _type_attribute_common:
                continue
            val = _type_attribute_common[attrkey].common(self, type1, type2)
            if not val is None:
                self.attr[attrkey] = val
        

    def toNumpy(self):
        """Returns dtype of a numpy container which
        can hold this type efficiently.
        """
        if(self.has_missing):
            return numpy.dtype(object)
        else:
            return numpy.dtype(self._dtype)

    def getSubType(self, subtype_id):
        """Returns subtype. For any type, this
        raises an error"""
        raise TypeError, "Expected subtypeable type, but found " + str(self)

    def __eq__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        return (self.__class__ is other.__class__ and 
                self.has_missing is other.has_missing)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(self.has_missing):
                return other.has_missing
            return True
        if(other.__class__ in self.__class__.__mro__):
            return True
        return False
    
    def __ge__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(other.has_missing):
                return self.has_missing
            return True
        if(self.__class__ in other.__class__.__mro__):
            return True
        return False

    def __gt__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            return self.has_missing and not other.has_missing
        if(self.__class__ in other.__class__.__mro__):
            return True
        return False
    
    def __lt__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            return not self.has_missing and other.has_missing
        if(other.__class__ in self.__class__.__mro__):
            return True
        return False

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.has_missing)

    def copy(self, **newattr):
        """Returns copy of this type"""
        res = self.__class__(has_missing=self.has_missing)
        res.attr = self.attr.copy()
        return res
    
    def __repr__(self):
        res = self.name
        if(self.has_missing):
            res += "$"
        return res
addType(TypeAny)#}}}

class TypeTuple(TypeAny):#{{{
    """Tuple or record type, having multiple values of 
       possibly different types"""
    name = "tuple"

    def __init__(self, has_missing=False, subtypes=(), 
                                 fieldnames=(), **attr):
        """
        Creates type object.

        Parameters
        ----------
        has_missing : bool, optional
        subtypes : tuple(type objects, i.e. subclasses of TypeUnknown), optional
        fieldnames : tuple(strings), optional
        scanner : scanner object, optional
        """

        assert isinstance(subtypes, tuple), \
                "The subtypes argument should be a tuple"

        assert all(isinstance(fieldname, str) for fieldname in fieldnames), \
                "Fieldnames should be strings"

        assert not fieldnames or len(fieldnames) == len(subtypes), \
            "Length of fieldnames should be equal to length of tuples (or empty"

        TypeAny.__init__(self, has_missing, **attr)

        self.subtypes = subtypes
        self.fieldnames = fieldnames
    
    @classmethod
    def commonType(cls, type1, type2):
        if(not type1.subtypes or not type2.subtypes or
            len(type1.subtypes) != len(type2.subtypes)):
            res =  cls(type1.has_missing or type2.has_missing)
        else:
            subtypes = [casts.castImplicitCommonType(lstype, rstype)
                      for lstype, rstype in zip(type1.subtypes, type2.subtypes)]
            fieldnames = type1.fieldnames                        
            res = cls(type1.has_missing or type2.has_missing, 
                       tuple(subtypes), fieldnames)
        res.setCommonAttributes(type1, type2)
        return res
    
    def toDefval(self):
        """Returns default value."""

        if(not "_defval" in self.__dict__):
            self._defval = tuple((subtype.toDefval() 
                                  for subtype in self.subtypes))
        return self._defval
    
    def getSubType(self, subtype_id):
        """Returns subtype identified by `id`

        Parameters
        ----------
        subtype_id : int, number of subtype

        Returns
        -------
        subtype, or unknown type if no subtypes given
        """
        if(self.subtypes):
            return self.subtypes[subtype_id]
        else:
            return unknown

    
    def __eq__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        return (self.__class__ is other.__class__ and 
                self.has_missing is other.has_missing and 
                self.subtypes == other.subtypes and 
                self.fieldnames == other.fieldnames)
 
    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(self.has_missing and not other.has_missing):
                return False
            if((not self.subtypes and other.subtypes) or (other.subtypes and
                len(self.subtypes) != len(other.subtypes))):
                return False
            if(not all([st <= ot for st, ot in 
                            zip(self.subtypes, other.subtypes)])):
                return False
            return True
        if(other.__class__ in self.__class__.__mro__):
            return True
        return False
    
    def __ge__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(other.has_missing and not self.has_missing):
                return False
            if((not other.subtypes and self.subtypes) or (self.subtypes and
                len(self.subtypes) != len(other.subtypes))):
                return False
            if(not all([st >= ot for st, ot in 
                            zip(self.subtypes, other.subtypes)])):
                return False
            return True
        if(self.__class__ in other.__class__.__mro__):
            return True
        return False

    def __gt__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(other.has_missing and not self.has_missing):
                return False
            if(not other.subtypes or (self.subtypes and 
                len(self.subtypes) != len(other.subtypes))):
                return False
            if(not all([st > ot for st, ot in 
                            zip(self.subtypes, other.subtypes)])):
                return False
            return True
        if(self.__class__ in other.__class__.__mro__):
            return True
        return False
    
    def __lt__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(self.has_missing and not other.has_missing):
                return False
            if(not self.subtypes or (other.subtypes and
                len(self.subtypes) != len(other.subtypes))):
                return False
            if(not all([st < ot for st, ot in 
                            zip(self.subtypes, other.subtypes)])):
                return False
            return True
        if(other.__class__ in self.__class__.__mro__):
            return True
        return False

    def __hash__(self):
        return (hash(self.__class__) ^ 
                hash(self.has_missing) ^ 
                hash(self.subtypes) ^
                hash(self.fieldnames))
    
    def copy(self, **newattr):
        """Returns copy of this type"""
        res = self.__class__(has_missing=self.has_missing,
                              subtypes=self.subtypes, 
                              fieldnames=self.fieldnames)
        res.attr = self.attr.copy()
        res.attr.update(newattr)
        return res
    

    def redim_var(self, deplen, var_adapt, redim_cache):
        return self.__class__(self.has_missing, 
                              tuple([subtype.redim_var(deplen, var_adapt, redim_cache) 
                                     for subtype in self.subtypes]), 
                              self.fieldnames, **self.attr)
        

    def __repr__(self):
        res = '(' 
        if(len(self.fieldnames) == len(self.subtypes)):
            res += ", ".join((fname + "=" + str(subtype) 
                for fname, subtype in zip(self.fieldnames, self.subtypes)))
        else:
            res += ", ".join((str(subtype) 
                    for subtype in self.subtypes))
        res += ')'
        if(self.has_missing):
            res += "$"
        return res
addType(TypeTuple)#}}}

class TypeDict(TypeTuple):#{{{
    name = "dict"

    def __repr__(self):
        res = '{' 
        if(len(self.fieldnames) == len(self.subtypes)):
            res += ", ".join((fname + "=" + str(subtype) 
                for fname, subtype in zip(self.fieldnames, self.subtypes)))
        else:
            res += ", ".join((str(subtype) 
                    for subtype in self.subtypes))
        res += '}'
        if(self.has_missing):
            res += "$"
        return res

addType(TypeDict)#}}}

#pylint: disable-msg=E1101
class TypeArray(TypeAny):#{{{
    """Type representing a collection of values, 
       possibly in an dimensional structure"""
    name = "array"

    def __init__(self, has_missing=False, dims=(), \
                       subtypes=(unknown,), **attr):
        """
        Creates type object.

        Parameters
        ----------
        has_missing : bool, optional
        dims : tuple(Dim's), optional
        subtypes : tuple(type object, i.e. subclass of TypeUnknown), optional
        scanner : scanner object, optional

        A dimid is a unique identifier for a dimension, helping the
        system to determine equality between dimensions. If there are
        no similarities with dimensions in other types, should be left empty.
        """
        assert isinstance(dims, tuple), \
                "Dims of an array should be a tuple"
        assert all([isinstance(dim, dimensions.Dim) for dim in dims]), \
                "Dims tuple should contain Dim objects"
        assert isinstance(subtypes, tuple) and len(subtypes) == 1, \
                "One subtype should be set for array type"
        if dims:
            dims[0].has_missing = has_missing

        self.subtypes = subtypes
        self.dims = dims
        TypeAny.__init__(self, has_missing, **attr)
    
    @classmethod
    def commonType(cls, type1, type2):
        if(not type1.dims or not type2.dims or
            len(type1.dims) != len(type2.dims)
            or type1.dims != type2.dims):
            return cls(has_missing=type1.has_missing or type2.has_missing)
        else:
            subtypes = [casts.castImplicitCommonType(lstype, rstype)
                        for lstype, rstype in zip(type1.subtypes, type2.subtypes)]
            dims = type1.dims 
            res = cls(has_missing=type1.has_missing or type2.has_missing, 
                       dims=dims, subtypes=tuple(subtypes))
        res.setCommonAttributes(type1, type2)
        return res

    def toDefval(self):
        """Returns default value."""

        subtype = self.subtypes[0]
        if(not self.dims):
            res = numpy.array([], dtype=subtype.toNumpy())
        else:
            shape = [dim.shape for dim in self.dims]
            for pos, sdim in enumerate(shape):
                if(sdim < 0):
                    shape[pos] = 0
            
            res = numpy.empty(self.dims, dtype=subtype.toNumpy())
            subdv = subtype.toDefval()
           
            #workaround numpy problem 
            #(cannot set sequence for a range)
            if(operator.isSequenceType(subdv)):
                flatres = res.ravel()
                for i in range(len(flatres)):
                    flatres[i] = subdv
            else:
                res[:] = subdv
        return res
    
    def getSubType(self, subtype_id=0):
        assert (subtype_id == 0), "Invalid subtype id given"
        if(self.subtypes):
            return self.subtypes[subtype_id]
        else:
            return unknown
    
    def __eq__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        return (self.__class__ is other.__class__ and 
                self.has_missing is other.has_missing and 
                self.subtypes == other.subtypes and 
                self.dims == other.dims)
 
    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(self.has_missing and not other.has_missing):
                return False
            if((not self.subtypes and other.subtypes) or
                len(self.subtypes) != len(other.subtypes)):
                return False
            if(not all([st <= ot for st, ot in 
                            zip(self.subtypes, other.subtypes)])):
                return False
            return all([d1.shape == d2.shape for d1, d2 in 
                                zip(self.dims, other.dims)])
        if(other.__class__ in self.__class__.__mro__):
            return True
        return False
    
    def __ge__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(other.has_missing and not self.has_missing):
                return False
            if((not other.subtypes and self.subtypes) or
                len(self.subtypes) != len(other.subtypes)):
                return False
            if(not all([st >= ot for st, ot in 
                            zip(self.subtypes, other.subtypes)])):
                return False
            return all([d1.shape == d2.shape for d1, d2 in 
                                    zip(self.dims, other.dims)])
        if(self.__class__ in other.__class__.__mro__):
            return True
        return False

    def __gt__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(other.has_missing and not self.has_missing):
                return False
            if(not other.subtypes or
                len(self.subtypes) != len(other.subtypes)):
                return False
            if(not all([st > ot for st, ot in 
                            zip(self.subtypes, other.subtypes)])):
                return False
            return all([d1.shape == d2.shape for d1, d2 in 
                                    zip(self.dims, other.dims)])
        if(self.__class__ in other.__class__.__mro__):
            return True
        return False
    
    def __lt__(self, other):
        if(isinstance(other, str)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(self.has_missing and not other.has_missing):
                return False
            if(not self.subtypes or
                len(self.subtypes) != len(other.subtypes)):
                return False
            if(not all([st < ot for st, ot in 
                            zip(self.subtypes, other.subtypes)])):
                return False
            return all([d1.shape == d2.shape for d1, d2 in 
                                    zip(self.dims, other.dims)])
        if(other.__class__ in self.__class__.__mro__):
            return True
        return False

    def __hash__(self):
        return (hash(self.__class__) ^ 
                hash(self.has_missing) ^ 
                hash(self.subtypes) ^ 
                hash(self.dims))
    
    def copy(self, **newattr):
        """Returns copy of this type"""
        res = self.__class__(has_missing=self.has_missing, dims=self.dims, 
                              subtypes=self.subtypes) 
        res.attr = self.attr.copy()
        res.attr.update(newattr)
        return res

    
    def redim_var(self, deplen, var_adapt, redim_cache):
        """Redim nested dimension, in response to change/removal of parent dim(s)
           deplen: distance from (nearest) parent dim that was changed/removed
           var_adapt: list with adaptions of dims.variable distances, w.r.t to parent dims.
                      E.g, nested dim dependent only on last parent dim uses var_adapt[0],
                      nested dim depnedent on last two, uses var_adapt[1], etc.
                      Parent dims are indicated by deplen
           redim_cache: store translation from old dims to new dims
        """
        stype = self.subtypes[0].redim_var(deplen + len(self.dims), var_adapt, redim_cache)
        ndims = []
        var_found = False
        for pos, dim in enumerate(self.dims):
            var_found = True
            if dim.variable - (deplen + pos) > 0:
                if(dim in redim_cache):
                    ndims.append(redim_cache[dim])
                else:
                    ndim = dim.copy(reid=True)
                    ndim.variable = max(ndim.variable + var_adapt[min(dim.variable - (deplen + pos) - 1, len(var_adapt)-1)],0)
                    ndims.append(ndim)
                    redim_cache[dim] = ndim
            else:
                ndims.append(dim)
        if(var_found or stype != self.subtypes[0]):
            self = self.copy()
            self.subtypes = (stype,)
            self.dims = tuple(ndims)
            return self
        else:
            return self

    def __repr__(self, unpack_depth=0):
        
        res = '[' + ",".join([str(dim) for dim in self.dims]) + ']'
        if(self.has_missing):
            res += "$"
        if(unpack_depth > 1):
            assert isinstance(self.subtypes, TypeArray), \
                   "Unpack depth of tuple is larger than number of " + \
                   "array subtypes. Found: " + str(self.subtypes[0])
            res += '<' + self.subtypes[0].__repr__(unpack_depth - 1)
        else:
            res += ':' + str(self.subtypes[0]) 
        return res
addType(TypeArray)#}}}

class TypeSet(TypeArray):#{{{
    name = "set"
    
    def __init__(self, has_missing=False, dims=(), subtypes=(unknown,), **attr):
        assert (isinstance(dims, tuple) and len(dims) == 1), \
                "Dimensions of a set should be a tuple of size 1"
        
        assert subtypes and len(subtypes) == 1, \
                 "Number of subtypes should be 1"
        
        subtypes = (type_attribute_freeze.freeze_protocol.freeze(subtypes[0]),)
        TypeArray.__init__(self, has_missing, dims, 
                                    subtypes, **attr)
   
    def __repr__(self):
        res = ""
        if(self.has_missing):
            res += "$"
        res += '{' + ",".join([str(dim) for dim in self.dims]) + '}'
        res += '<' + str(self.subtypes[0]) 
        return res

addType(TypeSet)#}}}

class TypeString(TypeArray):#{{{
    """String type (supports unicode strings)"""
    name = "string"
    _dtype = "U"
    _defval = u""
    
    def __init__(self, has_missing=False, dims=(), **attr):
        assert (isinstance(dims, tuple) and len(dims) == 1), \
            "Dimensions of a string should be a tuple of size 1"
        TypeArray.__init__(self, has_missing, dims, 
                                    (TypeChar(),), **attr)

    @classmethod
    def commonType(cls, type1, type2):
        if(type1.dims[0].shape == UNDEFINED or
            type2.dims[0].shape == UNDEFINED):
            shape = UNDEFINED
        else:
            shape = max(type1.dims[0].shape, type2.dims[0].shape)
        dim = dimensions.Dim(shape)
        res = cls(has_missing=type1.has_missing or type2.has_missing, 
                    dims=(dim,),)

        res.setCommonAttributes(type1, type2)
        return res

    def toNumpy(self):
        """Returns dtype of a numpy container which
           can hold this type efficiently."""

        if(self.dims[0].shape == UNDEFINED or self.has_missing or self.dims[0].shape > 32):
            return numpy.dtype(object)
        else:
            return numpy.dtype(self._dtype + str(self.dims[0].shape))
    
    def __eq__(self, other):
        return (self.__class__ is other.__class__ and 
                self.has_missing is other.has_missing and \
                self.dims == other.dims)

    def __hash__(self):
        return (hash(self.__class__) ^ 
                hash(self.has_missing) ^ 
                hash(self.dims))
    
    def copy(self, **newattr):
        """Returns copy of this type"""
        res = self.__class__(has_missing=self.has_missing, 
                              dims=self.dims)
        res.attr = self.attr.copy()
        res.attr.update(newattr)
        return res
    

    def __repr__(self):
        res =  self.name
        if(self.has_missing):
            res += "$"
        if(self.dims[0].shape >= 0):
            res += '[' + str(self.dims[0].shape) + ']'
        return res
addType(TypeString)#}}}

class TypeBytes(TypeString):#{{{
    """String type (only byte/ascii characters)"""
    name = "bytes"
    _dtype = "S"
    _defval = b""
addType(TypeBytes)#}}}

class TypeScalar(TypeAny):#{{{
    """Type representing atom-like values"""
    name = "scalar"

addType(TypeScalar)#}}}

class TypeChar(TypeScalar):#{{{
    """Type representing characters"""
    name = "char"
    
addType(TypeChar)#}}}

class TypeSlice(TypeScalar):#{{{
    """Type representing slices"""
    name = "slice"
    
addType(TypeSlice)#}}}

class TypeNumber(TypeScalar):#{{{
    """Type representing the number values"""
    name = "number"
    _defval = 0
    
    def __init__(self, has_missing=False, **attr):
        TypeScalar.__init__(self, has_missing, **attr)
    
    def copy(self):
        """Returns copy of this type"""
        return self.__class__(has_missing=self.has_missing, 
                               scanner=self.scanner)
addType(TypeNumber)#}}}

class TypeComplex(TypeNumber):#{{{
    """Complex number"""
    name = "complex"
    _defval = 0.0j
    
    def toNumpy(self):
        """Returns dtype of a numpy container which
        can hold this type efficiently.
        """
        return numpy.dtype(self._dtype)
    
addType(TypeComplex)#}}}

class TypeComplex128(TypeComplex):#{{{
    """Complex number, representable by machine doubles"""
    name = "complex128"
    _dtype = "complex128"
    _scalar = numpy.complex128
addType(TypeComplex128)#}}}

class TypeComplex64(TypeComplex128):#{{{
    """Complex number, representable by machine singles"""

    name = "complex64"
    _dtype = "complex64"
    _scalar = numpy.complex64
addType(TypeComplex64)#}}}

class TypeReal64(TypeComplex128):#{{{
    """Floating point number, representable by machine double"""
    name = "real64"
    _dtype = "float64"
    _scalar = numpy.float64
    _defval = 0.0
addType(TypeReal64)#}}}

class TypeReal32(TypeReal64, TypeComplex64):#{{{
    """Floating point number, representable by machine single"""
    name = "real32"
    _dtype = "float32"
    _scalar = numpy.float32
addType(TypeReal32)#}}}

class TypeInteger(TypeReal32):#{{{
    """Integer number"""
    name = "long"
    _dtype = "object"
    _minmax = (-numpy.inf, numpy.inf)
    
    @classmethod
    def getMinValue(cls):
        """Returns minimum integer that can be stored 
        with this type"""
        return cls._minmax[0]

    @classmethod
    def getMaxValue(cls):
        """Returns maximum integer that can be stored
        with this type"""
        return cls._minmax[1]
    
    def toNumpy(self):
        """Returns dtype of a numpy container which
        can hold this type efficiently.
        """
        if(self.has_missing):
            return numpy.dtype(object)
        else:
            return numpy.dtype(self._dtype)

addType(TypeInteger)#}}}

class TypeInt64(TypeInteger):#{{{
    """Integer number, 
    range -9,223,372,036,854,775,808 to +9,223,372,036,854,775,807"""
    name = "int64"
    _minmax = (-2**63, 2**63-1)
    _dtype = "int64"
    _scalar = numpy.int64
addType(TypeInt64)#}}}

class TypeInt32(TypeInt64):#{{{
    """Integer number,
    range -2,147,483,648 to +2,147,483,647"""

    name = "int32"
    _minmax = (-2**31, 2**31-1)
    _dtype = "int32"
    _scalar =  numpy.int32
addType(TypeInt32)#}}}

class TypeInt16(TypeInt32):#{{{
    """Integer number,
    range -32,768 to +32,767"""

    name = "int16"
    _minmax = (-2**15, 2**15-1)
    _dtype = "int16"
    _scalar = numpy.int16
addType(TypeInt16)#}}}

class TypeInt8(TypeInt16):#{{{
    """Integer number,
    range -128 to 127"""

    name = "int8"
    _minmax = (-2**7, 2**7-1)
    _dtype = "int8"
    _scalar = numpy.int8
addType(TypeInt8)#}}}

class TypeUnsignedInteger(TypeInteger):#{{{
    """Integer number (no negative integers)"""
    name = "ulong"
addType(TypeUnsignedInteger)#}}}

class TypeUInt64(TypeUnsignedInteger):#{{{
    """Integer number,
    range 0 to 18,446,744,073,709,551,615"""
    name = "uint64"
    _minmax = (0, 2**64-1)
    _dtype = "uint64"
    _scalar = numpy.uint64
addType(TypeUInt64)#}}}

class TypeUInt32(TypeUInt64, TypeInt64):#{{{
    """Integer number,
    range 0 to 4,294,967,295"""
    name = "uint32"
    _minmax = (0, 2**32-1)
    _dtype = "uint32"
    _scalar = numpy.uint32
addType(TypeUInt32)#}}}

class TypeUInt16(TypeUInt32, TypeInt32):#{{{
    """Integer number,
    range 0 to 65,535"""
    name = "uint16"
    _dtype = "uint16"
    _minmax = (0, 2**16-1)
    _scalar = numpy.uint16
addType(TypeUInt16)#}}}

class TypeUInt8(TypeUInt16, TypeInt16):#{{{
    """Integer number,
    range 0 to 255"""

    name = "uint8"
    _minmax = (0, 2**8-1)
    _dtype = "uint8"
    _scalar = numpy.uint8
addType(TypeUInt8)#}}}

class TypeBool(TypeUInt8, TypeInt8):#{{{
    """Integer number,
    range 0 to 1 (or False, True)"""

    name = "bool"
    _minmax = (0, 1)
    _dtype = "bool"
    _scalar = numpy.bool
    _defval = False
    
addType(TypeBool)#}}}




#maps numpy type to internal type
__objtype_map__ = {
numpy.dtype("object"):TypeAny,
numpy.dtype("int8"):TypeInt8,
numpy.dtype("int16"):TypeInt16,
numpy.dtype("int32"):TypeInt32,
numpy.dtype("int64"):TypeInt64,
numpy.dtype("uint8"):TypeInt8,
numpy.dtype("uint16"):TypeInt16,
numpy.dtype("uint32"):TypeInt32,
numpy.dtype("uint64"):TypeInt64,
numpy.dtype("float32"):TypeReal32,
numpy.dtype("float64"):TypeReal64,
numpy.dtype("complex64"):TypeComplex64,
numpy.dtype("complex128"):TypeComplex128}



NO_NAME = 0
NAME_FOUND = 1
INDIM = 2
INSUBTYPES = 3

NAME_FOUND_ARRAY = 2
IN_SUBTYPE = 2
EXIT_SUBTYPE = 3
IN_DIM = 4
OUT_DIM = 5
IN_SUBTYPE_DIM = 6

def createType(name):#{{{
    """Creates a type object from string representation.

    Parameters
    ----------
    name : str

    Examples
    --------
    createType("unicode")
    createType("array(int64)[]")
    createType("tuple(int64,unicode)")
    """

    if(not isinstance(name, str)):
        if(not isinstance(name, numpy.dtype)):
            name = numpy.dtype(name)
        
        if(name.char == 'S'):
            if(name.itemsize == 0):
                return TypeBytes(False)
            else:
                return TypeBytes(False, name.itemsize)
        elif(name.char == 'U'):
            if(name.itemsize == 0):
                return TypeString(False)
            else:
                return TypeString(False, name.itemsize / 4)
        elif(name in __objtype_map__):
            return __objtype_map__[name](False)
        else:
            raise TypeError,"Unknown type description: " + str(name)

    return _createType(name)[0]#}}}

def _createType(name, last_dims=()):#{{{
    begin_pos = 0
    state = NO_NAME
    
    typecls = None
    type_has_missing = False
    type_need_freeze = True

    subtypes = None
    dimlist = None
    fieldnames = None

    pos = -1
    name_len = len(name)
    fieldname = None
    while pos < (name_len - 1):
        pos += 1
        curchar = name[pos]
        if(curchar == "="):
            if(state == NO_NAME):
                fieldname = name[begin_pos:pos].strip()
                begin_pos = pos + 1
                continue
            assert False, "Encountered unexpected " + \
                                        curchar + " at: " + str(pos)
        elif(curchar == "["): #start array type
            if(state == NO_NAME):
                if(begin_pos == pos):
                    typecls = __typenames__["array"]
                else:
                    typecls = __typenames__[name[begin_pos:pos].strip()]
                state = INDIM
                dim_name = None
                dimlist = []
                begin_pos = pos + 1
                continue
            assert False, "Encountered unexpected " + \
                                        curchar + " at: " + str(pos)
        elif(curchar == "]"):
            if(state == INDIM):
                shape = name[begin_pos:pos].strip()
                if(shape):
                    dim_variable = 0
                    if shape == "~" or shape == "?":
                        dim_variable = 1
                        if(shape == "?"):
                            dim_shape = 1
                        else:
                            dim_shape = UNDEFINED
                    elif(shape == "*"):
                        dim_shape = UNDEFINED
                    else:
                        dim_shape = int(shape)
                        assert dim_shape > 0, "Shape should be larger " + \
                                        "than 0 (at pos: " + str(pos) + ")"
                else:
                    dim_variable = 0
                    dim_shape = UNDEFINED
                dimlist.append(dimensions.Dim(dim_shape, 
                                              dim_variable, did = dim_name))
                begin_pos = pos + 1
                state = NAME_FOUND 
                continue
            assert False, "Encountered unexpected " + \
                                        curchar + " at: " + str(pos)
        elif(curchar == ":"):
            if(state == INDIM):
                dim_name = name[begin_pos:pos].strip()
                begin_pos = pos + 1
                continue
            elif(state == NAME_FOUND):
                assert not name[begin_pos:pos].strip(), \
                    "Unexpected chars found before : " + "at pos: " + str(pos)
                subtypes = []
                begin_pos = pos + 1
                subtype, npos, fname = _createType(name[begin_pos:])
                subtypes.append(subtype)
                begin_pos = pos + npos + 1
                pos = begin_pos - 1
                continue
            assert False, "Encountered unexpected " + \
                                        curchar + " at: " + str(pos)
        elif(curchar == ","):
            if(state == INDIM):
                shape = name[begin_pos:pos].strip()
                if(shape):
                    if shape == "~" or shape == "?":
                        dim_variable = 1
                        if(shape == "?"):
                            dim_shape = 1
                        else:
                            dim_shape = UNDEFINED
                    elif(shape == "*"):
                        dim_shape = UNDEFINED
                    else:
                        dim_shape = int(shape)
                        assert dim_shape > 0, "Shape should be larger " + \
                                           "than 0 (at pos: " + str(pos) + ")"
                else:
                    dim_variable = 1
                    dim_shape = UNDEFINED
                begin_pos = pos + 1
                dimlist.append(dimensions.Dim(dim_shape, 
                                              dim_variable, did = dim_name)) 
                continue
            elif(state == INSUBTYPES):
                assert not name[begin_pos:pos].strip(), \
                    "Unexpected characters before ) at " + str(pos)
                begin_pos = pos + 1
                (subtype, npos, fname) = _createType(name[begin_pos:], last_dim)
                subtypes.append(subtype)
                fieldnames.append(fname)
                begin_pos = pos + npos + 1
                pos = begin_pos - 1
                continue
            elif(state == NO_NAME or state == NAME_FOUND):
                if(state == NO_NAME):
                    typecls = __typenames__[name[begin_pos:pos].strip()]
                    state = NAME_FOUND
                break
            assert False, "Encountered unexpected " + \
                                        curchar + " at: " + str(pos)
        elif(curchar == "$"):
            if(state == NO_NAME):
                typecls = __typenames__[name[begin_pos:pos].strip()]
                state = NAME_FOUND
            if(state == NAME_FOUND):
                begin_pos = pos + 1
                type_has_missing = True
                continue
            assert False, "Encountered unexpected " + \
                                        curchar + " at: " + str(pos)
        elif(curchar == "!"):
            if(state == NO_NAME):
                typecls = __typenames__[name[begin_pos:pos].strip()]
                state = NAME_FOUND
            if(state == NAME_FOUND):
                begin_pos = pos + 1
                type_need_freeze = False
                continue
            assert False, "Encountered unexpected " + \
                                        curchar + " at: " + str(pos)
        elif(curchar == "("):
            if(state == NO_NAME or state == NAME_FOUND):
                if(state == NO_NAME):
                    if(begin_pos == pos):
                        typecls = __typenames__["tuple"]
                    else:
                        typecls = __typenames__[name[begin_pos:pos].strip()]
                state = INSUBTYPES
                subtypes = []
                fieldnames = []
                last_dim = ()
                begin_pos = pos + 1
                
                (subtype, npos, fname) = _createType(name[begin_pos:], last_dim)
                subtypes.append(subtype)
                fieldnames.append(fname)
                begin_pos = pos + npos + 1
                pos = begin_pos - 1
                continue
            assert False, "Encountered unexpected " + \
                                        curchar + " at: " + str(pos)
                
        elif(curchar == ")"):
            if(state == INSUBTYPES):
                assert not name[begin_pos:pos].strip(), \
                    "Unexpected characters before ) at " + str(pos)
                begin_pos = pos + 1
                state = NAME_FOUND
                continue
            elif(state == NO_NAME or state == NAME_FOUND):
                if(state == NO_NAME):
                    typecls = __typenames__[name[begin_pos:pos].strip()]
                    state = NAME_FOUND
                break
        continue



    if(state == NO_NAME):
        if(len(name) == 0):
            return unknown
        else:
            typecls = __typenames__[name]


    params = {}
    if(not subtypes is None):
        params['subtypes'] = tuple(subtypes)
    if(not fieldnames is None):
        if(not all(fieldname is None for fieldname in fieldnames)):
            params['fieldnames'] = fieldnames
    if(not dimlist is None):
        params['dims'] = tuple(dimlist)
    if(not typecls is TypeUnknown):
        params['has_missing'] = type_has_missing
        if(type_need_freeze is False):
            params['need_freeze'] = type_need_freeze
        return (typecls(**params), pos, fieldname)
    else:
        assert (len(params) == 0), "Unknown with extra parameters?"
        return (unknown, pos, fieldname)#}}}

#### HELPER functions #########

def mostSpecializedTypesCls(typeclasses):
    """Takes list of typeclasses, and removes the 
    classes that also have a subclass of themself in the list"""
    basetypes = reduce(operator.__or__, 
                            [set(t.__mro__[1:]) for t in typeclasses])
    return [tcls for tcls in typeclasses  
                       if not tcls in basetypes]

def mostSpecializedTypes(typeobjs):
    """Takes list of typeobjects, and removes the 
    objects that also have a subclass of themself in the list"""
    basetypes = reduce(operator.__or__, 
                            [set(t.__class__.__mro__[1:]) for t in typeobjs])
    return [tobj for tobj in typeobjs  
                       if not tobj.__class__ in basetypes]


### sets ###
TypeNumbers = set(TypeNumber.getDescendantTypes())
TypeStrings = set(TypeString.getDescendantTypes())
TypeArrays = set(TypeArray.getDescendantTypes())



