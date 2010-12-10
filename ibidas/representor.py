import sys
import copy
import itertools
from collections import defaultdict

from query_graph import Node
from itypes import rtypes
from constants import *

_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"itypes", "dimensions")
_delay_import_(globals(),"wrappers","wrapper_py")
_delay_import_(globals(),"query_context")
_delay_import_(globals(),"engines")
_delay_import_(globals(),"repops")
_delay_import_(globals(),"repops_rel")
_delay_import_(globals(),"repops_dim")
_delay_import_(globals(),"repops_slice")
_delay_import_(globals(),"slices")

class Representor(Node):
    _state = 0                 #default value
    _slices = []               #default value

    def initialize(self, slices, state=RS_ALL_KNOWN):
        assert isinstance(slices, tuple), "slices should be a tuple"
        self._slices = slices
        if(state == RS_CHECK):
            if(slices):
                state |= RS_SLICES_KNOWN
            if(all([slice.type != rtypes.unknown for slice in slices])):
                state |= RS_TYPES_KNOWN
            state &= ~RS_CHECK
        self._state = state


    def checkState(self,filter=RS_SLICES_KNOWN):
        if(not ((self._state & filter) == filter)):
            slices = self._getResultSlices()
            self.initialize(slices,state=RS_ALL_KNOWN | RS_INFERRED)

    def __str__(self):
        self.checkState(filter=RS_ALL_KNOWN)
        table = []
        table.append(["Slices:", "Types:", "Dims:"])
        last_dim = ()
        for slice in self._slices:
            dim_str = []
            for pos, dim in enumerate(slice.dims):
                if len(last_dim) > pos and dim == last_dim[pos]:
                    dim_str.append(".")
                else:
                    dim_str.append(str(dim))
            last_dim = slice.dims
            dim_str = "<".join(dim_str)
            table.append([slice.name, str(slice.type), dim_str])

        res = util.create_strtable(table) + "\n"
        res += "Data: " + str(self())
        return res
    
    #def __repr__(self):
    #    return str(self.__class__)
    __repr__ = __str__

    def _axisF(self, name):
        return repops_slice.Project(self, name)

    def _axisA(self, name):
        return repops_slice.unpack_tuple(self, name)

    def _axisE(self, name):
        return repops_dim.unpack_array(self, name)

    def _axisD(self, name):
        #slices = self._active_dim_slice_dict[name] 
        return repops_slice.ProjectDim(self, name)
        

    def __getattr__(self, name):
        if(not name):
            return self

        try:
            axis_letter = name[0]
            #should be upper case
            if(not axis_letter.upper() == axis_letter or axis_letter == "_"):
                return repops_slice.Project(self,name)
                #prevent recursion
                slices = self.__dict__["_slices"]
                
                #assume default
                if(len(slices) == 1):
                    if(name == slices[0].name):
                        return self
                    axis_letter = "A" 
                else:
                    axis_letter = "F"
            else:
                name = name[1:]

            return getattr(self, "_axis" + axis_letter)(name)


        except AttributeError, error:
            #reraise attribute error as runtime error, 
            #so that python will not
            #attempt to find attribute in another way
            #(thereby raising a new exception)
            exc_class, exc, traceback = sys.exc_info()
            raise RuntimeError, RuntimeError(error.message), traceback
 
        raise AttributeError("No attribute with name: " + name + " found")

    def _getAttributeNames(self):
        return [slice.name for slice in self._slices]

    def copy(self):
        return wrapper_py.PyRepresentor(self._getResultSlices())

   
    def __reduce__(self):
        return (wrapper_py.PyRepresentor, (self._getResultSlices(),))

    def getType(self):
        if(len(self._slices) == 1):
            return self._slices[0].type
        else:
            return rtypes.TypeTuple(False, 
                    tuple([slice.type for slice in self._slices]), 
                    tuple([slice.name for slice in self._slices]))


    def __getitem__(self, condition):
        if(isinstance(condition, tuple)):
            ncond = len(condition)
            for pos, cond in enumerate(condition):
                if(isinstance(cond, context.Context)):
                    cond = context._apply(cond, self)
                self = repops_rel.rfilter(self, cond, pos - ncond)
            return self
            
        elif(isinstance(condition, context.Context)):
            condition = context._apply(condition, self)
        return repops_rel.rfilter(self, condition)
           
    def filter(self, condition, dim=False):
        if(isinstance(condition, context.Context)):
            condition = context._apply(condition, self)
        return repops_rel.rfilter(self, condition, dim) 

    def _getResultSlices(self, args={}):
        query = query_context.QueryContext(self, args)
        return engines.select_engine.run(query)

    def _copyquery(self):
        return copy.copy(self)

    def __call__(self, **args):
        res = self._getResultSlices(args)

        if(len(self._slices) == 1):
            return res[0].data
        else:
            return tuple([slice.data for slice in res])

    # Overloaded functions. The r* functions are implemented because you
    # want to the same behaviour no matter no which side the known object
    # resides. (a + b == b + a)
    # add ( + )
    def __add__(self, other):
        if(isinstance(other, context.Context)):
            return other.__radd__(self)
        elif(isinstance(other, repops.PlusPrefix)):
            pass
        return repops_rel.binop(self, other, '__add__')
    
    def __radd__(self, other):
        return repops_rel.binop(self, other, '__radd__')
    
    def __sub__(self, other):
        if(isinstance(other, context.Context)):
            return other.__radd__(self)
        elif(isinstance(other, repops.PlusPrefix)):
            pass
        return repops_rel.binop(self, other, '__sub__')
    
    def __rsub__(self, other):
        return repops_rel.binop(self, other, '__rsub__')

    # multiplication ( * )
    def __mul__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rmul__(self)
        elif(isinstance(other, repops.PlusPrefix)):
            pass
        return repops_rel.binop(self, other, '__mul__')
    
    def __rmul__(self, other):
        return repops_rel.binop(self, other, '__rmul__')
    
    # modulo ( % )
    def __mod__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rmul__(self)
        elif(isinstance(other, repops.PlusPrefix)):
            pass
        return repops_rel.binop(self, other, '__mod__')
    
    def __rmod__(self, other):
        return repops_rel.binop(self, other, '__rmod__')

    # division ( / )
    def __div__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rdiv__(self)
        elif(isinstance(other, str)):
            return repops_slice.slice_rename(self, other)
        elif(isinstance(other, tuple)):
            return repops_slice.slice_rename(self, *other)
        elif(isinstance(other, dict)):
            return repops_slice.slice_rename(self, **other)
        return repops_rel.binop(self, other, '__div__')
    
    def __rdiv__(self, other):
        return repops_rel.binop(self, other, '__rdiv__')
    
    def __floordiv__(self, other):
        if(isinstance(other, str)):
            return self.bm(other)
        else:
            raise NotImplementedError
   
    # and operator ( & )
    def __and__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rand__(self)
        elif(isinstance(other, repops.PlusPrefix)):
            pass
        return repops_rel.binop(self, other, '__and__')
    
    def __rand__(self, other):
        return repops_rel.binop(self, other, '__rand__')
    
    # or operator ( | )
    def __or__(self, other):
        if(isinstance(other, context.Context)):
            return other.__ror__(self)
        elif(isinstance(other, repops.PlusPrefix)):
            pass
        return repops_rel.binop(self, other, '__or__')

    def __ror__(self, other):
        return repops_rel.binop(self, other, '__ror__')
    
    # exclusive-or operator ( ^ )
    def __xor__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rxor__(self)
        elif(isinstance(other, repops.PlusPrefix)):
            pass
        return repops_rel.binop(self, other, '__xor__')

    def __rxor__(self, other):
        return repops_rel.binop(self, other, '__rxor__')

    # less-than ( < )
    def __lt__(self, other):
        if(isinstance(other, context.Context)):
            return other.__gt__(self)
        return repops_rel.binop(self, other, '__lt__')

    # less-than-or-equals ( <= )
    def __le__(self, other):
        if(isinstance(other, context.Context)):
            return other.__ge__(self)
        return repops_rel.binop(self, other, '__le__')

    # equals ( == )
    def __eq__(self, other):
        if(isinstance(other, context.Context)):
            return other.__eq__(self)
        return repops_rel.binop(self, other, '__eq__')

    # not-equals ( != )
    def __ne__(self, other):
        if(isinstance(other, context.Context)):
            return other.__ne__(self)
        return repops_rel.binop(self, other, '__ne__')

    # greater-than ( > )
    def __gt__(self, other):
        if(isinstance(other, context.Context)):
            return other.__lt__(self)
        return repops_rel.binop(self, other, '__gt__')

    # greater-than-or-equals ( >= )
    def __ge__(self, other):
        if(isinstance(other, context.Context)):
            return other.__le__(self)
        return repops_rel.binop(self, other, '__ge__')

    # plus prefix, used for table operators (i.e. ++, &+, etc.)
    def __pos__(self):
        return repops.PlusPrefix(self)
    
    def __invert__(self):
        return repops_rel.unary_op(self, "__invert__")
       
    def __abs__(self):
        return repops_rel.unary_op(self, "__abs__")
    
    def __neg__(self):
        return repops_rel.unary_op(self, "__neg__")

    def flat(self, dim_selector=-1):
        return repops_rel.flat(self, dim_selector)

    def group_by(self, *args, **kwargs):
        keep = kwargs.pop("keep", {})
        name = kwargs.pop("name", None)
        group_slice = self.get(*args, **kwargs)
        
        if(isinstance(keep,dict)):
            pass
        elif(isinstance(keep, list)):
            keep = {0:keep}
        else:
            keep = {0:[keep]}
            
        return repops_rel.Group(self, group_slice, keep, name)
   
    def match(self, other, condleft, condright, group=False):
        return repops_rel.match(self, other, condleft, condright, group=group)
    
    def join(self, other, cond=None, ldim=None, rdim=None):
        return repops_rel.join(self, other, cond, ldim, rdim)

    def rename(self, *names, **kwds):
        return repops_slice.slice_rename(self, *names, **kwds)
    
    def map(self, func, otype=rtypes.unknown, dim=None, *params, **kwds):
        return repops_rel.rmap(self, func, otype, dim, *params, **kwds)

    def sum(self):
        return repops_rel.rsum(self)
    def max(self):
        return repops_rel.rmax(self)
    def min(self):
        return repops_rel.rmin(self)
    def mean(self):
        return repops_rel.mean(self)
    def any(self):
        return repops_rel.rany(self)
    def all(self):
        return repops_rel.rall(self)
    def count(self):
        return repops_rel.count(self)

    def set(self):
        return repops_rel.rset(self)
    def array(self):
        return repops_dim.rarray(self)

    def list(self):
        return repops_dim.rlist(self)
    
    def tuple(self, to_python=False):
        return repops_slice.RTuple(self, to_python)

    def to_python(self):
        if(len(self._slices) > 1):
            self = self.tuple(to_python=True)

        while(self._slices[0].dims):
            self = self.list()

        return self()

    def sort(self, slice):
        return repops_rel.sort(self, self.get(slice))

    def get(self, *slices, **kwds):
        return repops_slice.Project(self,*slices,**kwds)

    def _getActiveDimDict(self):
        if(not hasattr(self, '_active_dim_cache')):
            res = defaultdict(set)
            for slice in self._slices:
                for dim in slice.dims:
                    res[dim.name].add(dim)
            self._active_dim_cache = res
        return self._active_dim_cache
    _active_dim_dict = property(_getActiveDimDict)

    def _getActiveSliceDict(self):
        if(not hasattr(self, '_active_slice_cache')):
            res = defaultdict(set)
            for slice in self._slices:
                res[slice.name].add(slice)
            self._active_slice_cache = res
        return self._active_slice_cache
    _active_slice_dict = property(_getActiveSliceDict)

    def _getActiveDimSliceDict(self):
        if(not hasattr(self, '_active_dim_slice_cache')):
            res = defaultdict(set)
            for slice in self._slices:
                for dim in slice.dims:
                    res[dim.name].add(slice)
            self._active_dim_slice_cache = res
        return self._active_dim_slice_cache
    _active_dim_slice_dict = property(_getActiveDimSliceDict)


    def _getActiveDimParentDict(self):
        if(not hasattr(self, '_active_dim_parent_cache')):
            res = defaultdict(set)
            for slice in self._slices:
                for pdim, cdim in zip(slice.dims, slice.dims[1:]):
                    res[cdim.name].add(pdim)
                if(slice.dims):
                    res[slice.dims[0].name].add(None)
                    res[None].add(slice.dims[-1])
            self._active_dim_parent_cache = res
        return self._active_dim_parent_cache
    _active_dim_parent_dict = property(_getActiveDimParentDict)        
    
    def _getActiveDimChildDict(self):
        if(not hasattr(self, '_active_dim_child_cache')):
            res = defaultdict(set)
            for slice in self._slices:
                for pdim, cdim in zip(slice.dims, slice.dims[1:]):
                    res[pdim.name].add(cdim)
                if(slice.dims):
                    res[slice.dims[-1].name].add(None)
                    res[None].add(slice.dims[0])
            self._active_dim_child_cache = res
        return self._active_dim_child_cache
    _active_dim_child_dict = property(_getActiveDimChildDict)        
    
    def _getActiveDimIdDict(self):
        if(not hasattr(self, '_active_dim_id_cache')):
            res = {}
            for slice in self._slices:
                for dim in slice.dims:
                    res[dim.id] = dim
            self._active_dim_id_cache = res
        return self._active_dim_id_cache
    _active_dim_id_dict = property(_getActiveDimIdDict)

    def _getActiveDimIdSliceDict(self):
        if(not hasattr(self, '_active_dim_id_slice_cache')):
            res = defaultdict(set)
            for slice in self._slices:
                for dim in slice.dims:
                    res[dim.id].add(slice)
            self._active_dim_id_slice_cache = res
        return self._active_dim_id_slice_cache
    _active_dim_id_slice_dict = property(_getActiveDimIdSliceDict)


    def _getActiveDimIdParentDict(self):
        if(not hasattr(self, '_active_dim_id_parent_cache')):
            res = defaultdict(set)
            for slice in self._slices:
                for pdim, cdim in zip(slice.dims, slice.dims[1:]):
                    res[cdim.id].add(pdim)
                if(slice.dims):
                    res[slice.dims[0].id].add(None)
                    res[None].add(slice.dims[-1])
            self._active_dim_id_parent_cache = res
        return self._active_dim_id_parent_cache
    _active_dim_id_parent_dict = property(_getActiveDimIdParentDict)        
    
    def _getActiveDimIdChildDict(self):
        if(not hasattr(self, '_active_dim_id_child_cache')):
            res = defaultdict(set)
            for slice in self._slices:
                for pdim, cdim in zip(slice.dims, slice.dims[1:]):
                    res[pdim.id].add(cdim)
                if(slice.dims):
                    res[slice.dims[-1].id].add(None)
                    res[None].add(slice.dims[0])
            self._active_dim_id_child_cache = res
        return self._active_dim_id_child_cache
    _active_dim_id_child_dict = property(_getActiveDimIdChildDict)       

