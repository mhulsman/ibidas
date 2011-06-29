import copy
import weakref

from ..utils import util
from ..constants import *
import itertools

_delay_import_(globals(),"dimpaths")


#pylint: disable-msg=E1101
dimid = util.seqgen().next
def getNewDimid():
    return dimid()
class Dim(object):
    """Class representing a dimension."""
    __slots__ = ['id', 'name', 'shape', 'dependent', 'has_missing','redim_cache','__weakref__']
   
    def __reduce__(self):
        return (Dim, (self.shape, self.dependent, self.has_missing, self.id, self.name))

    def __init__(self, shape, dependent=tuple(), has_missing=False, did=None, name=None):
        """Creates dimension class.

        :param shape: shape of dimension. Can be integer >= 0 or UNDEFINED
        :param dependent: tuple of bools, indicating on which parent dimension this dim is dependent
        :param has_missing: indicates if there are Missing values
        :param did:   id of dimension, optional
        """
        if(not did):
            self.id = getNewDimid()
        else:
            self.id = did

        assert isinstance(self.id, int), "ID should be a integer"

        if(name is None):
            self.name = "d" + str(self.id)
        else:
            assert name.lower() == name, "Dimension name should be lower case"
            self.name = name
        
        self.has_missing = has_missing
        
        assert (isinstance(dependent, tuple) and all([isinstance(elem,bool) for elem in dependent])), "Variable should be a tuple of bools"
        while(dependent and dependent[-1] is False): #remove False at end, as they do not contribute
            dependent = dependent[:-1]

        self.dependent = dependent
        self.shape = shape

    def isVariable(self):
        return len(self.dependent) > 0

    def __len__(self):
        return len(self.dependent)
    
    def __iter__(self):
        return self.dependent.__iter__()

    def _getRedimCache(self):
        try:
            return self.redim_cache
        except AttributeError:
            self.redim_cache = weakref.WeakValueDictionary()
            return self.redim_cache
            

    def removeDepDim(self, pos, elem_specifier):
        assert pos >= 0, "Position should not be smaller than 0"
        if(pos >= len(self.dependent)):
            return self

        key = (0, pos, elem_specifier)
        redim_cache = self._getRedimCache()
        if(not key in redim_cache):
            nself = self.copy(reid=self.dependent[pos])
            ndependent = self.dependent[:pos] + self.dependent[(pos + 1):]
            while(ndependent and ndependent[-1] is False):
                ndependent = ndependent[:-1]
            nself.dependent = ndependent
            redim_cache[key] = nself

        return redim_cache[key]
    
    def updateDepDim(self, pos, ndim):
        assert pos >= 0, "Position should not be smaller than 0"
        if(pos >= len(self.dependent)):
            return self

        if(not isinstance(ndim,tuple)):
            ndim = (ndim,)
        assert ndim, "No new dimensions specified"
        assert len(ndim) == 1, "Trap: updatedepdim with multiple dims"

        key = (1,pos, tuple([nd.id for nd in ndim]))
        redim_cache = self._getRedimCache()
        if(not key in redim_cache):
            nself = self.copy(reid=self.dependent[pos])
            if nself.dependent[pos]:
                ndependent = tuple([a or b for a,b in itertools.izip_longest(ndim[0].dependent, nself.dependent[(pos+1):],fillvalue=False)])
                nself.dependent = nself.dependent[:(pos + 1)] + ndependent
            redim_cache[key] = nself
        return redim_cache[key]
    
    def insertDepDim(self, pos, ndim):
        assert pos >= 0, "Position should not be smaller than 0"
        if(pos >= len(self.dependent)):
            return self

        redim_cache = self._getRedimCache()
        key = (2, pos, ndim.id)
        if(not key in redim_cache):
            nself = self.copy()
            nself.dependent = self.dependent[:pos] + (False,) + self.dependent[pos:]
            redim_cache[key] = nself
        return redim_cache[key]

    def changeDependent(self, dep, ndims):
        assert len(dep) == len(ndims), "Number of flags should be equal to number of dims when setting dependent dims"
        dep = tuple(dep)
        while(dep and dep[-1] is False):
            dep = dep[:-1]

        if(dep == self.dependent):
            return self

        redim_cache = self._getRedimCache()
        key = (3, tuple([ndim.id for d, ndim in zip(dep,ndims) if d]))

        if(not key in redim_cache):
            nself = self.copy(reid=True)
            nself.dependent = dep
            redim_cache[key] = nself
        nself = redim_cache[key]

        if(not nself.dependent == dep):
            nself = self.copy(reid=False)
            nself.dependent = dep

        return nself
       

    def copy(self, reid=False):
        """Returns a copy of this object"""
        res = copy.copy(self)
        if(reid):
            res.id = getNewDimid()
        return res

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and  self.id == other.id)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        if(self.dependent):
            if(self.shape == 1):
                res = self.name + ":?"
            else:
                res = self.name + ":~"
        elif(self.shape == UNDEFINED):
            res = self.name + ":*"
        else:
            res = self.name + ":" + str(self.shape)
        if(self.has_missing):
            res += "?"
        return res


    def merge(self, other):
        if self == other:
            return self
        if self.shape == other.shape and self.shape != UNDEFINED and self.name == other.name and not self.dependent and not other.dependent and not self.has_missing and not other.has_missing:
            return self
        redim_cache = self._getRedimCache()
        key = (self,other)
        if not key in redim_cache:
            if(self.shape == UNDEFINED or other.shape == UNDEFINED):
                rshape = UNDEFINED
            else:
                rshape = max(self.shape, other.shape)

            ndep = tuple([ldep or rdep for ldep, rdep in itertools.izip_longest(self.dependent, other.dependent,fillvalue=False)])

            if(self.name == other.name):
                nname = self.name
            else:
                nname = self.name + "_" + other.name
            
            ndim = Dim(rshape, ndep, self.has_missing or other.has_missing, name=nname)
            redim_cache[key] = ndim
        return redim_cache[key]

            

def toCommonDim(name, dims):
    shapes = set()
    dependent = set()
    for dim in dims:
        shapes.add(dim.shape)
        dependent.add(dim.dependent)
    shapes.discard(UNDEFINED)
    assert len(shapes) <= 1, "Dims with different shape cannot be cast to each other"
    if not shapes:
        shape = UNDEFINED
    else:
        shape = shapes.pop()

    assert len(dependent) == 1, "Dims with different dependencies cannot be cast to each other"
    dependent = dependent.pop()
    return Dim(shape, dependent, name=name)


        
        
