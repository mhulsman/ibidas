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

.. rubric:: Adding new types
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


.. rubric:: Hierarchy rules
    One should only inherit if the child type can only contain a subset 
    of the values of the parent type. 

    Operations on the parent type should be applicable on the child type 
    (but one can override them for more type-specific behaviour).

    Multiple inheritance is allowed, e.g. TypeUInt8(TypeUInt16, TypeInt16)

    It is currently not possible to add types within the hieararchy (e.g.
    a type inbetween TypeUnknown and any of the other types), 
    without modifying the source in this file or some hackish runtime class modification. 
"""


import platform
import copy
import numpy
import operator
from collections import defaultdict

from ..constants import *
from ..utils import util
from ..parser_objs import *
_delay_import_(globals(),"dimensions")
_delay_import_(globals(),"dimpaths")
_delay_import_(globals(),"casts")
_delay_import_(globals(),"type_attribute_freeze")
_delay_import_(globals(),"..utils.missing","Missing")

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

class Type(object):#{{{
    """Base type class. Represents the type of a data structure.
    """
    name = "?"
    _dtype = "object"
    _scalar = numpy.object
    _defval = None
    _reqRPCcon=True
    has_missing = True

    @classmethod
    def commonType(cls, type1, type2):
        """Returns a common supertype for type1 and type2"""
        return unknown

    def _requiresRPCconversion(self):
        return self._reqRPCcon

    def toNumpy(self):
        """Returns numpy dtype compatible with this type

        :rtype: numpy dtype
        """
        return numpy.dtype(self._dtype)

    def toScalar(self):
        """Returns numpy scalar classs compatible with this type
        
        :rtype: numpy scalar class
        """
        return self._scalar

    def toDefval(self):
        """Returns default value."""
        return self._defval
  
    def toMissingval(self):
        return Missing
    
    def hasMissingValInfo(self):
        try:
           self.toMissingval()
           return True
        except RuntimeError:
           return False

    def hasMissing(self):
        return self.has_missing

    def getName(self):
        """Returns base type name"""
        return self.name

    def _callSubtypes(self, methodname, *params, **kwds):
        """If has subtype, calls subtype with methodname and params

        :param methodname: Name of type method. Should return type object
        :param params: arguments for ``methodname``.
        :param kwds: Keyword arguments for ``methodname``
        """
        return self

    #dim changes
    def _removeDepDim(self, pos, elem_specifier):
        return self._callSubtypes("_removeDepDim",pos, elem_specifier)

    def _updateDepDim(self, pos, ndim):
        return self._callSubtypes("_updateDepDim",pos, ndim)

    def _insertDepDim(self, pos, ndim):
        return self._callSubtypes("_insertDepDim",pos, ndim)

    def _permuteDepDim(self, prevdims, permute_idxs):
        return self._callSubtypes("_permuteDepDim",prevdims, permute_idxs)

    #subtypes
    def getSubTypeNumber(self):
        """Returns number of subtypes of this type"""
        return 0

    def getSubType(self, subtype_id=0):
        """Returns subtype if this is a non-scalar type.
        Otherwise, raises TypeError. If subtype_id invalid, raised IndexError.

        :param subtype_id: id of subtype, default 0. Should be >= 0 and < :py:meth:`getSubTypeNumber`.
        :rtype: obj of class :py:class:`Type`"""
        return unknown
    
    #comparison
    def __eq__(self, other):
        if(isinstance(other, basestring)):
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

    def __repr__(self):
        return self.name
    
    def copy(self):
        """Returns copy of this type"""
        return self
   
    @classmethod
    def getDescendantTypes(cls):
        """Returns descendant type classes as list

        :rtype: list of descendant type classes"""
        if(not hasattr(cls, "_desc_types")):
            desc_types = [tcls.getDescendantTypes() 
                                for tcls in __typechildren__[cls]]
            #flatten list (concat each list in list)
            if(desc_types):
                desc_types = sum(desc_types, [])
            desc_types.append(cls)
            cls._desc_types = desc_types
        return cls._desc_types

class TypeUnknown(Type):
    """Unknown type represents the lack of information about
    the actual type. 

    As this type has only one possible state, 
    a module singleton is available through rtypes.unknown
    """
addType(TypeUnknown)#}}}

#unknown singleton
unknown = TypeUnknown()

class TypeAny(TypeUnknown):#{{{
    """Type which represents that any value is allowed"""

    name = "any"

    def __init__(self, has_missing=False):
        """
        Creates type object.

        :param has_missing: bool, optional
        """
        TypeUnknown.__init__(self)
        self.has_missing = has_missing
        
    @classmethod
    def commonType(cls, type1, type2):
        return cls(type1.has_missing or type2.has_missing)
   
    def toNumpy(self):
        """Returns dtype of a numpy container which
        can hold this type efficiently.
        """
        if(self.has_missing):
            return numpy.dtype(object)
        else:
            return numpy.dtype(self._dtype)
    
    def getSubType(self, subtype_id=0):
        """Returns subtype if this is a non-scalar type.
        Otherwise, raises TypeError. If subtype_id invalid, raised IndexError.

        :param subtype_id: id of subtype, default 0. Should be >= 0 and < :py:meth:`getSubTypeNumber`.
        :rtype: obj of class :py:class:`Type`"""
        raise TypeError, "Expected subtypeable type, but found " + str(self)

    def __eq__(self, other):
        if(isinstance(other, basestring)):
            other = createType(other)
        return (self.__class__ is other.__class__ and 
                self.has_missing is other.has_missing)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        if(isinstance(other, basestring)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(self.has_missing):
                return other.has_missing
            return True
        if(other.__class__ in self.__class__.__mro__):
            return True
        return False
    
    def __ge__(self, other):
        if(isinstance(other, basestring)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            if(other.has_missing):
                return self.has_missing
            return True
        if(self.__class__ in other.__class__.__mro__):
            return True
        return False

    def __gt__(self, other):
        if(isinstance(other, basestring)):
            other = createType(other)
        if(other.__class__ == self.__class__):
            return self.has_missing and not other.has_missing
        if(self.__class__ in other.__class__.__mro__):
            return True
        return False
    
    def __lt__(self, other):
        if(isinstance(other, basestring)):
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
        return copy.copy(self)

    def setHasMissing(self, value):
        if self.has_missing != value:
            res = self.copy()
            res.has_missing = value
        else:
            res = self
        return res

    def __repr__(self):
        res = self.name
        if(self.has_missing):
            res += "?"
        return res
addType(TypeAny)#}}}

class TypeTuple(TypeAny):#{{{
    """Tuple or record type, having multiple values of 
       possibly different types"""
    name = "tuple"

    def __init__(self, has_missing=False, subtypes=(), 
                       fieldnames=()):
        """
        Creates type object.

        :param has_missing: bool, optional
        :param subtypes: tuple(type objects, i.e. subclasses of TypeUnknown), optional
        :param fieldnames: tuple(strings), optional
        """

        assert isinstance(subtypes, tuple), \
                "The subtypes argument should be a tuple"

        assert all(isinstance(fieldname, basestring) for fieldname in fieldnames), \
                "Fieldnames should be strings"

        assert not fieldnames or len(fieldnames) == len(subtypes), \
            "Length of fieldnames should be equal to length of tuples (or empty"

        TypeAny.__init__(self, has_missing)
        self.subtypes = subtypes
        self.fieldnames = fieldnames
    
    @classmethod
    def commonType(cls, type1, type2):
        if(not type1.subtypes or not type2.subtypes or
            len(type1.subtypes) != len(type2.subtypes)):
            return cls(type1.has_missing or type2.has_missing)
        else:
            subtypes = [casts.castImplicitCommonType(lstype, rstype)
                      for lstype, rstype in zip(type1.subtypes, type2.subtypes)]
            if(False in subtypes):
                return False
            res = cls(type1.has_missing or type2.has_missing, tuple(subtypes), type1.fieldnames) 
        return res
    
    def toDefval(self):
        """Returns default value."""

        if(not "_defval" in self.__dict__):
            self._defval = tuple((subtype.toDefval() 
                                  for subtype in self.subtypes))
        return self._defval
   
    def getSubTypeNumber(self):
        """Returns number of subtypes of this type"""
        return len(self.subtypes)

    def getSubType(self, subtype_id=0):
        """Returns subtype if this is a non-scalar type.
        Otherwise, raises TypeError. If subtype_id invalid, raised IndexError.

        :param subtype_id: id of subtype, default 0. Should be >= 0 and < :py:meth:`getSubTypeNumber`.
        :rtype: obj of class :py:class:`Type`"""
        if(self.subtypes):
            return self.subtypes[subtype_id]
        else:
            return unknown

    
    def __eq__(self, other):
        if(isinstance(other, basestring)):
            other = createType(other)
        return (self.__class__ is other.__class__ and 
                self.has_missing is other.has_missing and 
                self.subtypes == other.subtypes and 
                self.fieldnames == other.fieldnames)
 
    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        if(isinstance(other, basestring)):
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
        if(isinstance(other, basestring)):
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
        if(isinstance(other, basestring)):
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
        if(isinstance(other, basestring)):
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
    
    def _callSubtypes(self, methodname, *params, **kwds):
        nsubtypes = tuple([getattr(subtype,methodname)(*params,**kwds) for subtype in self.subtypes])

        if(all([nsubtype is subtype for nsubtype,subtype in zip(nsubtypes,self.subtypes)])):
            nself = self
        else:
            nself = self.copy()
            nself.subtypes = nsubtypes
        return nself

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
            res += "?"
        return res
addType(TypeTuple)#}}}

class TypeRecordDict(TypeTuple):#{{{
    name = "record_dict"

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
            res += "?"
        return res

addType(TypeRecordDict)#}}}

class TypeIndexDict(TypeTuple):#{{{
    name = "index_dict"
    
    def __init__(self, has_missing=False, subtypes=(), 
                       fieldnames=("key","value")):
        """
        Creates type object.

        :param has_missing: bool, optional
        :param subtypes: tuple(type objects, i.e. subclasses of TypeUnknown), optional
        :param fieldnames: tuple(strings), optional
        """

        assert isinstance(subtypes, tuple), \
                "The subtypes argument should be a tuple"

        assert all(isinstance(fieldname, basestring) for fieldname in fieldnames), \
                "Fieldnames should be strings"

        assert not fieldnames or len(fieldnames) == len(subtypes), \
            "Length of fieldnames should be equal to length of tuples (or empty"

        assert len(subtypes) == 2, "IndexDict should have a key and value subtype"

        TypeAny.__init__(self, has_missing)
        self.subtypes = subtypes
        self.fieldnames = fieldnames

    def __repr__(self):
        res = 'dict(' 
        if(len(self.fieldnames) == len(self.subtypes)):
            res += ": ".join((fname + "=" + str(subtype) 
                for fname, subtype in zip(self.fieldnames, self.subtypes)))
        else:
            res += ": ".join((str(subtype) 
                    for subtype in self.subtypes))
        res += ')'
        if(self.has_missing):
            res += "?"
        return res

addType(TypeIndexDict)#}}}

#pylint: disable-msg=E1101
class TypeArray(TypeAny):#{{{
    """Type representing a collection of values, 
       possibly in an dimensional structure"""
    name = "array"

    def __init__(self, has_missing=False, dims=(), \
                       subtypes=(unknown,)):
        """
        Creates type object.

        :param has_missing: bool, optional
        :param dims: tuple(Dim's), optional
        :param subtypes: tuple(type object, i.e. subclass of TypeUnknown), optional

        A dimid is a unique identifier for a dimension, helping the
        system to determine equality between dimensions. If there are
        no similarities with dimensions in other types, should be left empty.
        """
        assert isinstance(dims, dimpaths.DimPath), \
                "Dims of an array should be a dimpath"
        assert len(dims) == 1, "Array should have one dimension"
        assert all([isinstance(dim, dimensions.Dim) for dim in dims]), \
                "Dims tuple should contain Dim objects"
        assert isinstance(subtypes, tuple) and len(subtypes) == 1, \
                "One subtype should be set for array type"
        
        has_missing = dims[0].has_missing or has_missing
        self.subtypes = subtypes
        self.dims = dims
        TypeAny.__init__(self, has_missing)
    
    @classmethod
    def commonType(cls, type1, type2):
        if(not type1.dims or not type2.dims or
            len(type1.dims) != len(type2.dims)):
            return TypeAny(type1.has_missing or type2.has_missins)
        else:
            subtypes = [casts.castImplicitCommonType(lstype, rstype)
                        for lstype, rstype in zip(type1.subtypes, type2.subtypes)]
            if(False in subtypes):
                return False
            ndims = []
            for ldim, rdim in zip(type1.dims, type2.dims):
                ndims.append(ldim.merge(rdim))

            dims = dimpaths.DimPath(*ndims)
            res = cls(has_missing=type1.has_missing or type2.has_missing, dims=dims, subtypes=tuple(subtypes)) 
        return res

    def setHasMissing(self, value):
        if value and not self.dims[0].dependent: #fixed dim, needs has_missing subtype if unpacked
            s = self.subtypes[0].setHasMissing(value)
        else:
            s = self.subtypes[0]
            
        if not self.has_missing == value or not s is self.subtypes[0]:
            self = self.copy()
            self.has_missing = value
            self.subtypes = (s,)
            
        return self


    def toMissingval(self):
        s = self.subtypes[0].toMissingval()
        if self.dims[0].dependent:
            res = numpy.array([],dtype=self.subtypes[0].toNumpy())
        else:
            if self.dims[0].shape == UNDEFINED:
                raise RuntimeError, "Cannot determine shape for missing value. Please cast dim " + str(self.dims[0])
            res = numpy.array([s] * self.dims[0].shape,dtype=self.toNumpy())
        return res

    def hasMissing(self):
        return self.subtypes[0].hasMissing()


    def toDefval(self):
        """Returns default value."""
        subtype = self.subtypes[0]
        shape = [dim.shape for dim in self.dims]
        for pos, sdim in enumerate(shape):
            if(sdim < 0):
                shape[pos] = 0
        
        res = numpy.empty(shape, dtype=subtype.toNumpy())
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
    

    def getSubTypeNumber(self):
        """Returns number of subtypes of this type"""
        return len(self.subtypes)

    def getSubType(self, subtype_id=0):
        """Returns subtype if this is a non-scalar type.
        Otherwise, raises TypeError. If subtype_id invalid, raised IndexError.

        :param subtype_id: id of subtype, default 0. Should be >= 0 and < :py:meth:`getSubTypeNumber`.
        :rtype: obj of class :py:class:`Type`"""
        assert (subtype_id == 0), "Invalid subtype id given"
        if(self.subtypes):
            return self.subtypes[subtype_id]
        else:
            return unknown
    
    def __eq__(self, other):
        if(isinstance(other, basestring)):
            other = createType(other)
        return (self.__class__ is other.__class__ and 
                self.has_missing is other.has_missing and 
                self.subtypes == other.subtypes and 
                self.dims == other.dims)
 
    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        if(isinstance(other, basestring)):
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
        if(isinstance(other, basestring)):
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
        if(isinstance(other, basestring)):
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
        if(isinstance(other, basestring)):
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
    
    def _callSubtypes(self, methodname, *params, **kwds):
        nsubtypes = tuple([getattr(subtype,methodname)(*params,**kwds) for subtype in self.subtypes])

        if(all([nsubtype is subtype for nsubtype,subtype in zip(nsubtypes,self.subtypes)])):
            nself = self
        else:
            nself = self.copy()
            nself.subtypes = nsubtypes
        return nself
   
    def _removeDepDim(self, pos, elem_specifier):
        nself = self._callSubtypes("_removeDepDim",pos - len(self.dims), elem_specifier)
        ndims = self.dims.removeDim(pos,elem_specifier)
        if(not ndims is self.dims):
            if(self is nself):
                nself = self.copy()
            nself.dims = ndims
        return nself
     
    def _updateDepDim(self, pos, ndim):
        nself = self._callSubtypes("_updateDepDim", pos - len(self.dims), ndim)
        ndims = self.dims.updateDim(pos,ndim)
        if(not ndims is self.dims):
            if(self is nself):
                nself = self.copy()
            nself.dims = ndims
        return nself

    def _insertDepDim(self, pos, ndim):
        nself = self._callSubtypes("_insertDepDim",pos - len(self.dims), ndim)
        ndims = self.dims.insertDim(pos, ndim)
        if(not ndims is self.dims):
            if(self is nself):
                nself = self.copy()
            nself.dims = ndims
        return nself

    def _permuteDepDim(self, prevdims, permute_idxs):
        nself = self._callSubtypes("_permuteDepDim", prevdims + self.dims, permute_idxs)
        ndims = self.dims.permuteDims(permute_idxs, prevdims=prevdims)
        if(not ndims is self.dims):
            if(self is nself):
                nself = self.copy()
            nself.dims = ndims
        return self

    def __repr__(self, unpack_depth=0):
        
        res = '[' + ",".join([str(dim) for dim in self.dims]) + ']'
        if(self.has_missing):
            res += "?"
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

    def __init__(self, has_missing=False, dims=(), subtypes=(unknown,)):

        assert (isinstance(dims, dimpaths.DimPath) and len(dims) == 1), \
                "Dimensions of a set should be a dimpath of size 1"

        assert subtypes and len(subtypes) == 1 and isinstance(subtypes,tuple), \
                 "Number of subtypes should be 1"
        
        TypeArray.__init__(self, has_missing, dims, 
                                 subtypes)
    def __repr__(self):
        res = ""
        if(self.has_missing):
            res += "?"
        res += '{' + ",".join([str(dim) for dim in self.dims]) + '}'
        res += '<' + str(self.subtypes[0]) 
        return res
    
    def hasMissing(self):
        return self.has_missing

    def toMissingval(self):
        return frozenset()

addType(TypeSet)#}}}

class TypeString(TypeArray):#{{{
    """String type (supports unicode strings)"""
    name = "string"
    _dtype = "U"
    _defval = u""
    _reqRPCcon=False
    
    def __init__(self, has_missing=False, dims=()):
        assert (isinstance(dims, dimpaths.DimPath) and len(dims) == 1), \
            "Dimensions of a string should be a dimpath of size 1"
        TypeArray.__init__(self, has_missing, dims, 
                                    (TypeChar(),))

    @classmethod
    def commonType(cls, type1, type2):
        if(type1.dims[0].shape == UNDEFINED or
            type2.dims[0].shape == UNDEFINED):
            shape = UNDEFINED
        else:
            shape = max(type1.dims[0].shape, type2.dims[0].shape)
        dim = dimensions.Dim(shape)
        res = cls(has_missing=type1.has_missing or type2.has_missing, 
                  dims=dimpaths.DimPath(dim))
        return res

    def toNumpy(self):
        """Returns dtype of a numpy container which
           can hold this type efficiently."""

        return numpy.dtype(object)
        #f(self.dims[0].shape == UNDEFINED or self.has_missing or self.dims[0].shape > 32):
        #   return numpy.dtype(object)
        #lse:
        #   return numpy.dtype(self._dtype + str(max(self.dims[0].shape,1)))
    
    def __eq__(self, other):
        return (self.__class__ is other.__class__ and 
                self.has_missing is other.has_missing and \
                self.dims[0].shape == other.dims[0].shape)

    def __hash__(self):
        return (hash(self.__class__) ^ 
                hash(self.has_missing) ^ 
                hash(self.dims[0].shape))

    def toMissingval(self):
        return Missing
    
    def hasMissing(self):
        return self.has_missing
   
    def __repr__(self):
        res =  self.name
        if(self.has_missing):
            res += "?"
        if(self.dims[0].shape >= 0):
            res += '[' + str(self.dims[0].shape) + ']'
        return res


    def toDefval(self):
        """Returns default value."""
        return ""

addType(TypeString)#}}}

class TypeBytes(TypeString):#{{{
    """String type (only byte/ascii characters)"""
    name = "bytes"
    _dtype = "S"
    _defval = b""
addType(TypeBytes)#}}}

class TypePickle(TypeBytes):#{{{
    name = "pickle"
addType(TypePickle)#}}}

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
    
    def __init__(self, has_missing=False):
        TypeScalar.__init__(self, has_missing)
    
addType(TypeNumber)#}}}

class TypeComplex(TypeNumber):#{{{
    """Complex number"""
    name = "complex"
    _defval = 0.0j
    
    def toNumpy(self):
        """Returns dtype of a numpy container which
        can hold this type efficiently.
        """
        if(self.has_missing):
            return numpy.dtype(object)
        else:
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
    _reqRPCcon=False
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
    _reqRPCcon=True
    
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




def createType(name, dimpos=0, refdims=[], env={}):#{{{
    """Creates a type object from string representation.

    :param name: str

    Format:
        Description of type formats

    Examples:
        >>> createType("unicode")

        >>> createType("array(int64)[]")

        >>> createType("tuple(int64,unicode)")

    """

    if(not isinstance(name, basestring)):
        if(not isinstance(name, numpy.dtype)):
            name = numpy.dtype(name)
        
        if(name.char == 'S'):
            if(name.itemsize == 0):
                dim = dimensions.Dim(UNDEFINED, (True,) * dimpos, False) 
                return TypeBytes(dims=dimpaths.DimPath(dim))
            else:
                dim = dimensions.Dim(name.itemsize, (True,) * dimpos, False) 
                return TypeBytes(dims=dimpaths.DimPath(dim))
        elif(name.char == 'U'):
            if(name.itemsize == 0):
                dim = dimensions.Dim(UNDEFINED, (True,) * dimpos, False) 
                return TypeString(dims=dimpaths.DimPath(dim))
            else:
                usize = numpy.array("test").dtype.itemsize / 4
                dim = dimensions.Dim(name.itemsize / usize, (True,) * dimpos, False) 
                return TypeString(dims = dimpaths.DimPath(dim))
        elif(name in __objtype_map__):
            return __objtype_map__[name](False)
        else:
            raise TypeError,"Unknown type description: " + str(name)

    return _createType(name, dimpos, refdims, env)#}}}

class TypeStringScanner(GenericScanner):#{{{
    def tokenize(self, input):
        self.rv = []
        GenericScanner.tokenize(self,input)
        return self.rv

    def t_whitespace(self,s):
        r' \s+ '
        pass

    def t_name(self,s):
        r' [a-zA-Z_][a-zA-Z_\d]* '
        t = Token(type='name',attr=s)
        self.rv.append(t)

    def t_integer(self,s):
        r' \d+ '
        t = Token(type="integer",attr=int(s))
        self.rv.append(t)

    def t_symbol(self,s):
        r' \= | \$ | \? | \. | \{ | \} | \< | \[ | \] | \( | \) | \, | \* | \~ | \! | \& | \# | \@ | \: | \^ | \| | \' '
        t = Token(type=s)
        self.rv.append(t)#}}}
    
class TypeStringParser(GenericParser):#{{{
    def __init__(self, start="typenest"):
        GenericParser.__init__(self, start)

    def p_param_1(self,args):
        '   param ::= @ name '
        return AST(type="param",kids=args[1:])

    def p_tmatcher(self, args):
        """
            tmatcher ::= typenest
            tmatcher ::= ^ typenest
        """
        if len(args) > 1:
            return AST(type="tmatcher",kids=args)
        else:
            return args[0]

    def p_tmatcher_list(self, args):
        """
            type_orlist ::= tmatcher
            type_orlist ::= type_orlist | tmatcher
        """
        return AST(type="type_orlist",kids=args[:1] + args[2:])

    def p_type_match_0(self, args):
        """
            type_andlist ::= type_orlist
            type_andlist ::= type_andlist & type_orlist
        """
        return AST(type="type_andlist", kids=args[:1] + args[2:])
   
    def p_type_andlist_1(self, args):
        """
            typenest ::= # type_andlist #
        """
        return AST(type="typenest", kids=args[1:2])

    def p_dmatcher(self, args):
        """
            dmatcher ::= nameddim
            dmatcher ::= ^ nameddim
        """
        if len(args) > 1:
            return AST(type="dmatcher",kids=args)
        else:
            return args[0]

    def p_dmatcher_list(self, args):
        """
            dim_orlist ::= dmatcher
            dim_orlist ::= dim_orlist | dmatcher
        """
        return AST(type="dim_orlist",kids=args[:1] + args[2:])
    
    def p_dim_match_0(self, args):
        """
            dim_andlist ::= dim_orlist
            dim_andlist ::= dim_andlist & dim_orlist
        """
        return AST(type="dim_andlist", kids=args[:1] + args[2:])

    def p_dim_andlist(self, args):
        """
            nameddim ::= # dim_andlist #
        """
        return AST(type="nameddim", kids=args[1:2])

    def p_param_call(self, args):
        """
            param ::= param ( typelist )
            param ::= param ( dimlist )
        """
        return AST(type="param_call", kids=[args[0], args[2]])

    def p_type_1(self,args):
        """
            type ::= param
            type ::= name 
            type ::= ?
        """
        return AST(type="createtype",kids=args[:1])
    
    def p_type_2(self,args):
        """
            type ::= type $
            type ::= type ?
            type ::= type !
        """
        return AST(type="hasmissing", kids=args)
    
    def p_type_2b(self,args):
        """
            type ::= type '
        """
        return AST(type="strict", kids=args)
   
    def p_type_3(self,args):
        ' type ::= type [ dimlist ] '
        return AST(type="dims",kids=(args[0], args[2]))
    
    def p_type_3b(self,args):
        ' type ::= type [ ] '
        return AST(type="dims",kids=(args[0],))

    def p_type_4(self,args):
        ' type ::= type ( typelist ) '
        return AST(type="subtypes",kids=(args[0], args[2]))
    
    def p_type_6(self,args):
        ' type ::= [ dimlist ] '
        return AST(type="dims",kids=(
            AST(type="createtype", kids=(Token(type="name",attr="array"),)), args[1]))
    
    def p_type_7(self,args):
        ' type ::= { dimlist } '
        return AST(type="dims",kids=(
            AST(type="createtype", kids=(Token(type="name",attr="set"),)), args[1]))
    
    def p_type_8(self,args):
        '''
            typeelem ::= typenest
            typeelem ::= name = typenest
        '''
        if len(args) == 1:
            return args[0]
        else:
            return AST(type="typeelem",kids=(args[2], args[0]))
    
    def p_type_9(self,args):
        ' type ::= ( typelist ) '
        return AST(type="subtypes",kids=(
            AST(type="createtype", kids=(Token(type="name",attr="tuple"),)), args[1]))
    
    def p_type_9b(self,args):
        ' type ::= { typelist } '
        return AST(type="subtypes",kids=(
            AST(type="createtype", kids=(Token(type="name",attr="record_dict"),)), args[1]))

    
    def p_typenest_2(self,args):
        """
        typenest ::= type
        typenest ::= typenest < type
        typenest ::= typenest : type
        typenest ::= typenest < typenest
        typenset ::= typenest : typenest
        """
        return AST(type="typenest",kids=args)

    def p_var_1(self,args):
        ''' 
            var ::= .
            var ::= *
        '''
        return args[0]
        
    def p_varlist_1(self,args):
        ''' 
            varlist ::= var
        '''
        return AST(type="varlist",kids=args[:1])
    
    def p_varlist_2(self,args):
        ''' 
            varlist ::= varlist var
        '''
        return AST(type="varlist",kids=args)


    def p_dim_1(self,args):
        ''' 
            dim ::= integer 
            dim ::= ~
            dim ::= varlist
            dim ::= param
        '''
        return AST(type="createdim",kids=args)


    def p_nameddim_1(self,args):
        ''' nameddim ::= name : dim '''
        return AST(type="namedim",kids=(args[2], args[0]))
    
    def p_nameddim_2(self,args):
        ''' nameddim ::= dim '''
        return args[0]
    
    def p_nameddim_3(self,args):
        ''' nameddim ::= name '''
        return AST(type="namedim",kids=(
            AST(type="createdim", kids=(Token(type="inherit"),)), args[0]))
   
    def p_nameddim_4(self,args):
        ''' 
            nameddim ::= nameddim  $
            nameddim ::= nameddim  ?
            nameddim ::= nameddim  !
        '''
        return AST(type="hasmissing",kids=args)

    def p_dim_6(self, args):
        """
            dimelem ::= nameddim
            dimelem ::= ?
            dimelem ::= ? ?
        """
        if(len(args) > 1):
            ntoken = Token(type="??")
        else:
            ntoken = args[0]
        return ntoken

    def p_dimlist_1(self,args):
        ''' 
            dimlist ::= dimelem 
            dimlist ::= dimlist , dimelem 
        '''
        return AST(type="dimlist",kids=args[:1] + args[2:])

    def p_typelist_1(self,args):
        ''' typelist ::= typeelem 
            typelist ::= typelist , typeelem '''
        return AST(type="typelist", kids=(args[:1] + args[2:]))#}}}

class TypeStringASTRewriterPass1(GenericASTRewriter):#{{{
    def process(self, tree):
        self.dim_annot = {}
        return (self.postorder(tree), self.dim_annot)
    
    def collapse_list(listname):
        def func(self, node):
            if(len(node.kids) > 1):
                if node.kids[0].type == listname:
                    node.kids = tuple(node.kids[0].kids) + (node.kids[1],)
                return node 
            else:
                return node.kids[0]
        return func
    n_type_orlist = collapse_list("type_orlist")
    n_dim_orlist = collapse_list("dim_orlist")
    n_type_andlist = collapse_list("type_andlist")
    n_dim_andlist = collapse_list("dim_andlist")
    n_typelist = collapse_list("typelist")

    def n_dimlist(self, node):
        if(node.kids[0].type == "dimlist"):
            node.kids = tuple(node.kids[0].kids) + (self.annotateDim(node.kids[1]),)
        else:
            node.kids = (self.annotateDim(node.kids[0]),)
        return node 
  
    def n_varlist(self, node):
        if(node.kids[0].type == "varlist"):
            node.kids = tuple(node.kids[0].kids) + (self.processVar(node.kids[1]),)
        else:
            node.kids = (self.processVar(node.kids[0]),)
        return node 

    def processVar(self,node):
        if(node.type == '.'):
            return True
        elif(node.type == '*'):
            return False
        else:
            raise RuntimeError, "Unexpected character as dim var"

    def n_typenest(self,node):
        if(len(node.kids) > 1):
            if node.kids[2].type == "typenest":
                node.packed = node.kids[0].packed + [node.kids[1].type == "<"] + node.kids[2].packed
                node.kids = tuple(node.kids[0].kids) + tuple(node.kids[2].kids)
            else:
                node.packed = node.kids[0].packed + [node.kids[1].type == "<"]
                node.kids = tuple(node.kids[0].kids) + (node.kids[2],)
        else:
            node.packed = []
            node.kids = (node.kids[0],)
        return node
    
    def n_createtype(self, node):
        node.kids = [node.kids[0], None, None, None, None, False] #hasmissing, name, dims, subtypes, strict
        return node
  
    def n_createdim(self, node):
        node.kids = [node.kids[0], None, None, False] #hasmissing, name, strict
        return node

    def n_hasmissing(self, node):
        if(node.kids[0].type == "createtype"):
            node.kids[0].kids[1] = node.kids[1].type in ("?","$")
        elif(node.kids[0].type == "createdim"):
            node.kids[0].kids[1] = node.kids[1].type in ("?","$")
        else:
            raise RuntimeError, "Invalid AST!"
        return node.kids[0]
    
    def n_strict(self, node):
        if(node.kids[0].type == "createtype"):
            node.kids[0].kids[5] = True
        elif(node.kids[0].type == "createdim"):
            node.kids[0].kids[3] = True
        else:
            raise RuntimeError, "Invalid AST!"
        return node.kids[0]
   
    def n_dims(self, node):
        assert node.kids[0].type == "createtype", "Invalid AST!"
        if(len(node.kids) > 1):
            assert node.kids[1].type == "dimlist", "Invalid AST!"
            node.kids[0].kids[3] = node.kids[1]
        return node.kids[0]

    def n_subtypes(self, node):
        assert node.kids[0].type == "createtype", "Invalid AST!"
        assert node.kids[1].type == "typelist", "Invalid AST!"
        node.kids[0].kids[4] = node.kids[1]
        return node.kids[0]
    
    def n_namedim(self, node):
        if(node.kids[0].type == "createdim"):
            node.kids[0].kids[2] = node.kids[1].attr
        else:
            return node
        return node.kids[0]
    
    def n_typeelem(self, node):
        if(node.kids[0].type == "createtype"):
            node.kids[0].kids[2] = node.kids[1].attr
        else:
            return node
        return node.kids[0]
    
    def annotateDim(self,node):
        if not node.type == "createdim":
            return node
       
        depshape, has_missing, name, strict = node.kids
        if(name is None):
            return node

        if(depshape.type == "varlist"):
            dependent = depshape.kids
            while(dependent and dependent[-1] is False):
                dependent = dependent[:-1]
            dependent = tuple(dependent)
            shape = UNDEFINED
        elif(depshape.type == "integer"):
            dependent = tuple()
            shape = depshape.attr
        elif(depshape.type == "~"):
            dependent = "~"
            shape = UNDEFINED
        elif(depshape.type == "inherit"):
            dependent = None
            shape = None
        else:
            raise RuntimeError, "Invalid AST!"

        if(name in self.dim_annot):
            ahas_missing, adependent, ashape = self.dim_annot[name]
            if(not ashape is None):
                if(not shape is None and shape != ashape):
                    raise RuntimeError, "Similar named dim: " + name + " with different shape: " + str(shape) + ", " + str(ashape)
                shape = ashape

            if(not adependent is None):
                if(not dependent is None and dependent != adependent):
                    raise RuntimeError, "Similar named dim: " + name + "  with different dependent dims: " + str(dependent) + ", " + str(adependent)
                dependent = adependent

            if(not ahas_missing is None):
                if(not has_missing is None and has_missing != ahas_missing):
                    raise RuntimeError, "Similar named dim: " + name + "  with different has_missing flag: "  + str(has_missing) + ", " + str(ahas_missing)
                has_missing = ahas_missing
        
        self.dim_annot[name] = (has_missing, dependent, shape)
        return node#}}}

class TypeStringASTRewriterPass2(GenericASTRewriter):#{{{
    def process(self, tree):
        ntree = self.postorder(tree)
        return ntree
        
    def n_typenest(self,node):
        kids = list(node.kids)
        while(len(kids) > 1):
            right = kids.pop()
            assert right.type == "createtype", "Invalid AST!"
            assert kids[-1].type == "createtype", "Invalid AST!"
            kids[-1].kids[4]= AST(type="typelist",kids=(right,))
        
        return kids[0]#}}}

class TypeStringASTInterpreter(object):#{{{
    def __init__(self, dim_annot, refdims, env={}):
        self.env = env
        self.dim_annot = dim_annot
        self.dims = dict([(dim.name,dim) for dim in refdims])
    
    def visit(self, node, dimpos=0):
        name = 'n_' + str(node.type)
        if hasattr(self, name):
            func = getattr(self, name)
            return func(node, dimpos)
        else:
            raise RuntimeError, "Cannot find method to process: " + name

    def n_createtype(self, node, dimpos=0):
        assert node.type == "createtype", "Cannot create type from this specification!"
      
        typetoken, has_missing, name, dimlist, subtypelist, strict = node.kids

        if has_missing is None:
            has_missing = False

        if(typetoken.type == '?'):
            typename = "?"
        else:
            typename = typetoken.attr
        if(typename not in __typenames__):
            raise RuntimeError, "Unknown type name: " + str(typename)
        typecls = __typenames__[typename]
       
        
        kwargs = {}
        if has_missing:
            kwargs['has_missing'] = True

        if(not dimlist is None):
            
            dims = dimpaths.DimPath(*[self.visit(dimnode, dimpos + pos) for pos, dimnode in enumerate(dimlist.kids)])
            dimpos += len(dims)
            kwargs['dims'] = dims
        elif(issubclass(typecls,TypeArray)):
            dims = dimpaths.DimPath(dimensions.Dim(UNDEFINED,(True,) * dimpos))
            dimpos += 1
            kwargs['dims'] = dims
        
        if not subtypelist is None:       
            subtypes = tuple([self.visit(subtypenode, dimpos) for subtypenode in subtypelist.kids])
            kwargs['subtypes'] = subtypes

        if(issubclass(typecls,TypeTuple)):
            fieldnames = []
            for pos, subtypenode in enumerate(subtypelist.kids):
                if subtypenode.type == "createtype":
                    name = subtypenode.kids[2]
                elif subtypenode.type == "typeelem":
                    name = subtypenode.kids[1].attr
                else:
                    name = None
                if name is None:
                    fieldnames.append("f" + str(pos))
                else:
                    fieldnames.append(util.valid_name(name))
            kwargs['fieldnames'] = tuple(fieldnames)

        if 'dims' in kwargs and len(kwargs['dims']) > 1:
            assert typecls == TypeArray, "Multidimensional types only allowed with arrays"
            dims = kwargs['dims'][::-1]
            kwargs['dims'] = dims[:1]
            subtype = typecls(**kwargs)
            for pos in xrange(1,len(dims)):
                subtype = TypeArray(dims=dims[pos:(pos+1)], subtypes=(subtype,))
            return subtype
        else:
            return typecls(**kwargs)

    def n_typeelem(self, node, dimpos):
        return self.visit(node.kids[0],dimpos)

    def n_param(self, node,dimpos):
        if not node.kids[0].attr in self.env:
            raise RuntimeError, "Cannot find variable '" + str(node.kids[0].attr) + "' in given variables"
        return self.env[node.kids[0].attr]

    def n_createdim(self, node, dimpos):
        assert node.type == "createdim", "Cannot create dim from this specification!"
        dimnode, has_missing, name, strict = node.kids
        if has_missing is None:
            has_missing = False
        
        if(dimnode.type == "varlist"):
            dependent = tuple(dimnode.kids)
            shape = UNDEFINED
        elif(dimnode.type == "integer"):
            dependent = tuple()
            shape = dimnode.attr
        elif(dimnode.type == "~"):
            dependent = (True,) * dimpos
            shape = UNDEFINED
        elif(dimnode.type == "inherit"):
            dependent = None
            shape = None
        else:
            raise RuntimeError, "Invalid AST!"
      
        if(not name is None):
            assert name in self.dim_annot, "Unannotated named dim found!"
            if(not name in self.dims):
                has_missing,dependent,shape = self.dim_annot[name]
                if(dependent is None):
                    dependent = tuple()
                elif(dependent == "~"):
                    dependent = (True,) * dimpos
                    
                if(shape is None):
                    shape = UNDEFINED
                self.dims[name] = dimensions.Dim(shape,dependent,has_missing, name=name) 
                    
            dim = self.dims[name]
            if(len(dim.dependent) > dimpos):
                raise RuntimeError, "Dim: " + name + " has too many dependent dims: " + str(len(dim.dependent)) + " (max: " + str(dimpos) + ")"
            if dim.shape != shape and not shape is None and not shape == UNDEFINED and not dim.shape == UNDEFINED:
                raise RuntimeError, "Dim: " + name + " dimension unequal to known dim"

            return dim


        return dimensions.Dim(shape,dependent,has_missing, name=name) #}}}

def _createType(name, dimpos=0, refdims=[], env={}):
    scanner = TypeStringScanner()
    tokens = scanner.tokenize(name)

    parser = TypeStringParser()

    tree = parser.parse(tokens)
    
    #print tree
    rewriter1 = TypeStringASTRewriterPass1()
    tree, dim_annotation = rewriter1.process(tree)
    #print tree, dim_annotation
    rewriter2 = TypeStringASTRewriterPass2()
    tree = rewriter2.process(tree)
    #print "2: " + str(tree)
    return TypeStringASTInterpreter(dim_annotation, refdims, env).visit(tree, dimpos)


class TypeStringMatchASTInterpreter(object):
    def process(self, tree, dims, matchtype, env, dim_annotation):
        self.packdepth = 0
        self.error_message = ""
        self.env = env
        self.dim_annotations = dim_annotation
        res = self.n_typenest(tree,matchtype,dims)
        return (res, self.packdepth, self.error_message, self.env)
   
    def setError(self, message):
        self.error_message = message
    def unsetError(self):
        self.error_message = ""
    
    def visit(self, node, matchers):
        name = 'n_' + str(node.type)
        if hasattr(self, name):
            func = getattr(self, name)
            if isinstance(matchers,tuple):
                mres = []
                for matcher in matchers:
                    mres.append(func(node, matcher))
                res = any([x for x,y in mres])
                ms = sum([y for x,y in mres],())
            else:
                res, ms = func(node, matchers)
            return (res, ms)
        else:
            raise RuntimeError, "Cannot find method to process: " + name

    def paramvisit(self, node, dims):
        name = 'p_' + str(node.type)
        if hasattr(self, name):
            func = getattr(self, name)
            return func(node, dims)
        else:
            raise RuntimeError, "Cannot find method to process: " + name

    def n_typenest(self, node, matchtype, dims=None):
        assert node.type == "typenest", "Invalid AST!"
        packlist = node.packed + [False]
        for packed, kid in zip(packlist[::-1], node.kids[::-1]):
            if packed:
                res, dims = self.paramvisit(kid, dims) 
                if res is False:
                    return (False,())
            else:
                if self.packdepth > 0:
                    raise RuntimeError, "Cannot intermix packed and non-packed types (unpacked: <, packed: :)"

        dims = None
        matchtypes = matchtype
        for pos, (packed, matcher) in enumerate(zip(packlist, node.kids)):
            if packed:
                continue
            res, matchtypes = self.visit(matcher, matchtypes)      
            if res is False:
                return (False,())
        return (True,matchtypes)

    def subtype(self, matchtypes):
        if isinstance(matchtypes, tuple):
            res = []
            for matchtype in matchtypes:
                if hasattr(matchtype, 'subtypes') and len(matchtype.subtypes) == 1:
                    res.append(matchtype.subtypes[0])
            res = tuple(res)
        else:
            if hasattr(matchtypes, 'subtypes') and len(matchtypes.subtypes) == 1:
                res = matchtypes.subtypes
            else:
                res = ()
        return res

    def p_createtype(self, node, dims):
        if not dims:
            self.setError("Asked to pack dimension, but no dimension found")
            return False, None
        if  node.kids[0].attr == "array":
            raise RuntimeError, "Only array type allowed in packed type matcher"
        
        dimlist = node.kids[3]
        if any([d.type == "??" for d in dimlist.kids]):
            raise RuntimeError, "No variable dimension lists (??) allowed in packed type matcher"

        res, rdims = self.visit(dimlist, (dims[-len(dimlist):],))
        self.packdepth += len(dimlist)
        return (res, dims[:-len(dimlist)])
    
    def n_param(self, node, matchtype):
        name = node.kids[0].attr
        if name in self.env and  self.env[name] != matchtype:
            self.setError("Matches of multiple recurrences of variable: " + name + " are not equal")
            return False
        self.env[name] = matchtype
        return (True, self.subtype((matchtype,)))

    def n_type_orlist(self, node, matchtype):
        results = [self.visit(matcher, matchtype) for matcher in node.kids]
        res = any([x for x, m in results])
        ms = sum([m for x, m in results if x],())
        if res is True:
            self.unsetError()
        return (res, ms)
    
    def n_type_andlist(self, node, matchtype):
        results = [self.visit(matcher, matchtype) for matcher in node.kids]
        res = all([x for x, m in results])
        ms = sum([m for x, m in results],())
        return (res, ms)

    def n_tmatcher(self, node, matchtype):
        if node.kids[0].type == "^":
            res,ms = self.visit(node.kids[1], matchtype)
            res = not res
            if res is False:
                self.setError("Type " + str(matchtype) + " matches negative constraint")
            else:
                self.unsetError()
            return (res, ms)
        else:
            return self.visit(node.kids[0], matchtype)


    def n_createtype(self, node, matchtype):
        typetoken, has_missing, name, dimlist, subtypelist, strict = node.kids

        if(typetoken.type == '?'):
            return (True,())
        elif(typetoken.type == 'param'):
            res, ms = self.visit(typetoken, matchtype) 
        else:
            typename = typetoken.attr
            
            if(typename not in __typenames__):
                raise RuntimeError, "Unknown type name: " + str(typename)
            typecls = __typenames__[typename]
        
            if not isinstance(matchtype, typecls):
                self.setError("Type class mismatch for: " + str(matchtype) + ", expected: " + typecls.name)
                return (False,())

        
        if not has_missing is None and not matchtype.has_missing is has_missing:
            self.setError("Type: " + str(matchtype) + " has incorrect has_missing state")
            return (False,())
                
        if not dimlist is None:
            if not hasattr(matchtype, "dims"):
                self.setError("Required type with dimensions, but found: " + str(matchtype))
                return (False,())
            res,md = self.visit(dimlist, (matchtype.dims,))
            if res is False:
                return (False,())
        
        if not subtypelist is None:
            if not hasattr(matchtype, "subtypes"):
                self.setError("Required type with subtypes, but found: " + str(matchtype))
                return (False,())
            if subtypelist.type == "typelist":
                if not len(subtypelist.kids) == len(matchtype.subtypes):
                    self.setError("Incorect number of subtypes, expected: " + str(len(subtypelist.kids)) + " but found: " + str(len(matchtype.subtypes)))
                    return (False,())
                for subtype, kid in zip(matchtype.subtypes, subtypelist.kids):
                    res,ms = self.visit(kid, subtype)
                    if res is False:
                        return (False,())
            elif subtypelist.type == "createtype":
                if not len(matchtype.subtypes) == 1:
                    self.setError("Incorect number of subtypes, expected 1 but found: " + str(len(matchtype.subtypes)))
                    return False
                res,ms = self.visit(subtypelist, matchtype.subtypes)
                if res is False:
                    return (False,())
            else:
                raise RuntimeError, "Unexpected subtypelist in AST"

        return (True, self.subtype(matchtype))

    def n_typeelem(self, node, matchtype):
        return self.visit(node.kids[0], matchtype)

    def n_createdim(self, node, dim):
        if not isinstance(dim, dimensions.Dim):
            self.setError("Expected dimension, but found none")
            return (False,())

        dimnode, has_missing, name, strict = node.kids
        if not has_missing is None and not dim.has_missing is has_missing:
            self.setError("Dimension: " + str(dim) + " has incorrect has_missing state")
            return (False,())
        
        if not name is None:
            if isinstance(self.dim_annotations[name], dimensions.Dim):
                if not self.dim_annotations[name] == dim:
                    self.setError("Dimensions with same name: " + name + " in matcher do not match in type for " + str(self.dim_annotations[name]) + " and " + str(dim))
                    return (False,())
            self.dim_annotations[name] = dim

        if(dimnode.type == "varlist"):
            dependent = tuple(dimnode.kids)
            while(dependent and dependent[-1] is False):
                dependent = dependent[:-1]
            if not dim.dependent == dependent:
                self.setError("Dimenension " + str(dim) + " does have incorrect dependence struture")
                return (False,())
        elif(dimnode.type == "integer"):
            if not dim.shape == dimnode.attr:
                self.setError("Dimenension " + str(dim) + " does have incorrect shape (" + str(dim.shape) + " instead of " + str(dimnode.attr) + ")")
                return (False,())
        elif(dimnode.type == "~"):
            if not dim.dependent:
                self.setError("Dimenension " + str(dim) + " should be variable but is not")
                return (False,())
        else:
            raise RuntimeError, "Invalid AST!"

        return (True,())

    def n_dimlist(self, node, dims):
        return self.match(node.kids, dims)
    
    def n_nameddim(self, node, dims):
        return self.visit(node.kids[0], (dims,))
    
    def n_dim_orlist(self, node, dims):
        xdims = (dims,)
        results = [self.visit(matcher, xdims) for matcher in node.kids]
        res = any([x for x, m in results])
        ms = sum([m for x, m in results if x],())
        if res is True:
            self.unsetError()
        return (res, ms)
    
    def n_dim_andlist(self, node, dims):
        xdism = (dims,)
        results = [self.visit(matcher, xdims) for matcher in node.kids]
        res = all([x for x, m in results])
        ms = sum([m for x, m in results],())
        return (res, ms)

    def match(self, dim_matchers, dims):
        if not dim_matchers and not dims:
            return (True,())
        elif not dims:
            self.setError("Expected dimension, but found none")
            return (False,())
        elif not dim_matchers:
            self.setError("Expected no dimension, but found one")
            return (False,())

        if dim_matchers[0].type == "?":
            if len(dims) == 0:
                self.setError("Expected dimension, but found none")
                return (False,())
            return self.match(dim_matchers[1:], dims[1:])
        elif dim_matchers[0].type == '??':
            mres = [self.match(dim_matchers[1:],dims[pos:])  for pos in xrange(len(dims) + 1)]
            res = any([x for x,y in mres])
            if res is True:
                self.unsetError()
            return (res,())
        else:
            res, md = self.visit(dim_matchers[0], dims[0])
            if res is False:
                return (False,())
            return self.match(dim_matchers[1:], dims[1:])


def matchType(name, matchtype, env=None, dims=None):
    scanner = TypeStringScanner()
    tokens = scanner.tokenize(name)

    parser = TypeStringParser()

    tree = parser.parse(tokens)
    #print '1: ', tree  
    rewriter1 = TypeStringASTRewriterPass1()
    tree, dim_annotation = rewriter1.process(tree)
    #print '2: ', tree, dim_annotation

    if env is None:
        env = {}
    return TypeStringMatchASTInterpreter().process(tree, dims, matchtype, env, dim_annotation)

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
TypeAll = set(TypeAny.getDescendantTypes())
TypeNumbers = set(TypeNumber.getDescendantTypes())
TypeStrings = set(TypeString.getDescendantTypes()) - set([TypePickle])
TypeArrays = set(TypeArray.getDescendantTypes())
TypeIntegers = set(TypeInteger.getDescendantTypes())
TypeReals = set(TypeReal64.getDescendantTypes()) - TypeIntegers

if(platform.architecture()[0] == "32bit"):
    TypePlatformInt = TypeInt32
else:
    TypePlatformInt = TypeInt64
__typenames__["int"] = TypePlatformInt    
