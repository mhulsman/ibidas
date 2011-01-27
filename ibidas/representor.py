import sys
import copy
import itertools
from collections import defaultdict

from query_graph import Node
from itypes import rtypes
from constants import *

_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"itypes", "dimensions","dimpaths")
_delay_import_(globals(),"wrappers","wrapper_py")
_delay_import_(globals(),"query_context")
_delay_import_(globals(),"engines")
_delay_import_(globals(),"repops")
_delay_import_(globals(),"repops_multi")
_delay_import_(globals(),"repops_dim")
_delay_import_(globals(),"repops_slice")
_delay_import_(globals(),"repops_funcs")
_delay_import_(globals(),"slices")



class NewDim(object):
    def __init__(self,name=None):
        self.name = name
    def __call__(self,name=None):
        return NewDim(name)
newdim = NewDim()


class Representor(Node):
    _state = 0                 #default value
    _slices = []               #default value

    def _initialize(self, slices, state=RS_ALL_KNOWN):
        assert isinstance(slices, tuple), "slices should be a tuple"
        self._slices = slices
        if(state == RS_CHECK):
            if(slices):
                state |= RS_SLICES_KNOWN
            if(all([slice.type != rtypes.unknown for slice in slices])):
                state |= RS_TYPES_KNOWN
            state &= ~RS_CHECK
        self._state = state
        
        #name checking
        slicenames = set()
        for slice in slices:
            if slice.name in slicenames:
                raise RuntimeError, "More than one slice with name: " + slice.name
            slicenames.add(slice.name)
            


    def checkState(self,filter=RS_SLICES_KNOWN):
        if(not ((self._state & filter) == filter)):
            slices = self._getResultSlices()
            self._initialize(slices,state=RS_ALL_KNOWN | RS_INFERRED)

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
        if(name == ""):
            name = None
        return repops_slice.UnpackTuple(self, name)

    def _axisE(self, name):
        if(name == ""):
            name = None
        return repops_dim.UnpackArray(self, name)

    def _axisD(self, name):
        #slices = self._active_dim_slice_dict[name] 
        return repops_slice.ProjectDim(self, name)

    def _axisB(self, name):
        return repops_slice.ProjectBookmark(self,name)

    def __getattr__(self, name):
        if(not name):
            return self

        try:
            axis_letter = name[0]
            #should be upper case
            if(not axis_letter.upper() == axis_letter or axis_letter == "_"):
                return repops_slice.Project(self,name)
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
        if not source._state & RS_SLICES_KNOWN:
            return []
        else:
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
    Rtype=property(fget=getType)

    def getDepth(self):
        return max([len(slice.dims) for slice in self._slices])
    Rdepth=property(fget=getDepth)
    

    def __getitem__(self, condition):
        if(not isinstance(condition, tuple)):
            condition = (condition,)
        
        #add dimensions first
        ipos = 0
        for pos, cond in enumerate(condition):
            if(isinstance(cond,NewDim)):
                self = repops_dim.InsertDim(self,pos + ipos, cond.name)
            elif(cond is Ellipsis):
                ulength = len(dimpaths.uniqueDimPath([s.dims for s in self._slices]))
                rem_length = len([c for c in condition[(pos + 1):] if not isinstance(cond,NewDim)])
                newnextpos = ulength - rem_length
                curnextpos = pos + ipos + 1
                ipos += newnextpos - curnextpos #skip some dims
        
        #next, perform filtering in backwards order
        #note: cannot use Ellipsis in condition, will perform Ellipsis == elem,
        #which wille xecute as query and fail/be slow
        if(any([elem is Ellipsis for elem in condition])):  
            ncond = len(dimpaths.uniqueDimPath([s.dims for s in self._slices]))- 1
        else:
            ncond = len(condition) - 1

        for pos, cond in enumerate(condition[::-1]):
            if(isinstance(cond, context.Context)):
                cond = context._apply(cond, self)
            if(isinstance(cond,slice) and cond.start is None and 
                                                cond.stop is None and 
                                                cond.step is None):
                pass
            elif(cond is Ellipsis):
                ncond = len(condition) - 1
            elif(isinstance(cond,NewDim)):
                pass
            elif(len(condition) == 1):
                self = repops_multi.Filter(self, cond)
            else:
                self = repops_multi.Filter(self, cond, ncond - pos)
        return self
           
    def filter(self, condition, dim=None):
        if(isinstance(condition, context.Context)):
            condition = context._apply(condition, self)
        return repops_multi.Filter(self, condition, dim) 

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
        return repops_funcs.Add(self, other)
    
    def __radd__(self, other):
        return repops_funcs.Add(other, self)
    
    def __sub__(self, other):
        if(isinstance(other, context.Context)):
            return other.__radd__(self)
        return repops_funcs.Subtract(self, other)
    
    def __rsub__(self, other):
        return repops_funcs.Subtract(other, self)

    # multiplication ( * )
    def __mul__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rmul__(self)
        return repops_funcs.Multiply(self, other)
    
    def __rmul__(self, other):
        return repops_funcs.Multiply(other, self)
    
    # modulo ( % )
    def __mod__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rmod__(self)
        elif(isinstance(other, str)):
            return repops_dim.DimRename(self, other)
        elif(isinstance(other, tuple)):
            return repops_dim.DimRename(self, *other)
        elif(isinstance(other, dict)):
            return repops_dim.DimRename(self, **other)
        return repops_funcs.Modulo(self, other)
    
    def __rmod__(self, other):
        return repops_funcs.Modulo(other, self)

    # division ( / )
    def __div__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rdiv__(self)
        elif(isinstance(other, str)):
            return repops_slice.SliceRename(self, other)
        elif(isinstance(other, tuple)):
            return repops_slice.SliceRename(self, *other)
        elif(isinstance(other, dict)):
            return repops_slice.SliceRename(self, **other)
        return repops_funcs.Divide(self, other)
    
    def __rdiv__(self, other):
        return repops_funcs.Divide(other, self)
    
    def __floordiv__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rfloordiv__(self)
        elif(isinstance(other, str)):
            return repops_slice.Bookmark(self, other)
        elif(isinstance(other, tuple)):
            return repops_slice.Bookmark(self, *other)
        elif(isinstance(other, dict)):
            return repops_slice.Bookmark(self, **other)
        return repops_funcs.FloorDivide(self, other)
    
    def __rfloordiv__(self,other):
        return repops_funcs.FloorDivide(other, self)
        

    def __pow__(self,other):
        if(isinstance(other, context.Context)):
            return other.__rpow__(self)
        return repops_funcs.Power(self, other)
    
    def __rpow__(self, other):
        return repops_funcs.Power(other,self)

    # and operator ( & )
    def __and__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rand__(self)
        return repops_funcs.And(self, other)
    
    def __rand__(self, other):
        return repops_funcs.And(other, self)
    
    # or operator ( | )
    def __or__(self, other):
        if(isinstance(other, context.Context)):
            return other.__ror__(self)
        return repops_funcs.Or(self, other)

    def __ror__(self, other):
        return repops_funcs.Or(other, self)
    
    # exclusive-or operator ( ^ )
    def __xor__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rxor__(self)
        return repops_funcs.Xor(self, other)

    def __rxor__(self, other):
        return repops_funcs.Xor(other, self)

    # less-than ( < )
    def __lt__(self, other):
        if(isinstance(other, context.Context)):
            return other.__gt__(self)
        return repops_funcs.Less(self, other)

    # less-than-or-equals ( <= )
    def __le__(self, other):
        if(isinstance(other, context.Context)):
            return other.__ge__(self)
        return repops_funcs.LessEqual(self, other)

    # equals ( == )
    def __eq__(self, other):
        if(isinstance(other, context.Context)):
            return other.__eq__(self)
        return repops_funcs.Equal(self, other)

    # not-equals ( != )
    def __ne__(self, other):
        if(isinstance(other, context.Context)):
            return other.__ne__(self)
        return repops_funcs.NotEqual(self, other)

    # greater-than ( > )
    def __gt__(self, other):
        if(isinstance(other, context.Context)):
            return other.__lt__(self)
        return repops_funcs.Greater(self, other)

    # greater-than-or-equals ( >= )
    def __ge__(self, other):
        if(isinstance(other, context.Context)):
            return other.__le__(self)
        return repops_funcs.GreaterEqual(self, other)

    # plus prefix, used for table operators (i.e. ++, &+, etc.)
    def __pos__(self):
        return repops.PlusPrefix(self)
    
    def __invert__(self):
        return repops_funcs.Invert(self)
       
    def __abs__(self):
        return repops_funcs.Abs(self)
    
    def __neg__(self):
        return repops_funcs.Negative(self)

    def __nonzero__(self):
        if(len(self._slices) != 1):
            raise RuntimeError, "Cannot determine True/False status of a multi-slice representor"

        slice = self._slices[0]
        if(len(slice.dims) > 1):
            nself = repops_dim.FlatAll(self)
        else:
            nself = self

        if(len(slice.dims) > 0):
            nself = repops_funcs.All(nself)
        
        res = nself()
        return bool(res)

    def cast(self, *newtypes, **kwds):
        return repops_slice.SliceCast(self, *newtypes, **kwds)

    def transpose(self,permute_idxs=(1,0)):
        return repops_dim.PermuteDims(self,permute_idxs)

    def flat(self, dim=-1,name=None):
        return repops_dim.Flat(self, name=name,dim=dim)

    def flatall(self, name=None):
        return repops_dim.FlatAll(self,name=name)

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
            
        return repops_multi.Group(self, group_slice, keep, name)
   
    def match(self, other, condleft, condright, group=False):
        return repops_multi.match(self, other, condleft, condright, group=group)
    
    def join(self, other, cond=None, ldim=None, rdim=None):
        return repops_multi.join(self, other, cond, ldim, rdim)

    def rename(self, *names, **kwds):
        """Rename slices.

        :param names: names without keywords. Number of names should match number of slices.
        :param kwds: names with keywords, e.g.  f0="genes". Number does *not* have to match number of slices.

        Examples::
            
            >>> na = a.rename("genes","scores")
            >>> na = a.rename(f0 = "genes", f1 = "scores")
        
        Shortcut:
            One can use the division operation to rename slices.

            * tuple:
                Number of names in tuple should match number of slices.

                >>> a/("genes", "scores")

            * dict:
                Similar to keywords.

                >>> a/{"f0": "genes"}

            * str:
                Can only be used if representor object consists of single slice.

                >>> a.f0/"genes"
        """ 
        return repops_slice.SliceRename(self, *names, **kwds)
    
    def dim_rename(self, *names, **kwds):
        return repops_dim.DimRename(self, *names, **kwds)
  
    def bookmark(self, *names, **kwds):
        return repops_slice.Bookmark(self, *names, **kwds)
        pass

    def map(self, func, otype=rtypes.unknown, dim=None, *params, **kwds):
        return repops_multi.rmap(self, func, otype, dim, *params, **kwds)

    def sum(self):
        return repops_multi.rsum(self)
    def max(self):
        return repops_multi.rmax(self)
    def min(self):
        return repops_multi.rmin(self)
    def mean(self):
        return repops_multi.mean(self)
    def any(self):
        return repops_multi.rany(self)
    def all(self):
        return repops_multi.rall(self)
    def count(self):
        return repops_multi.count(self)

    def set(self):
        return repops_multi.rset(self)

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

    def sort(self, *slices, **kwargs):
        if(slices or kwargs):
            sortsource = self.get(*slices,**kwargs)
            return repops_multi.sort(self, sortsource)
        else:
            return repops_multi.sort(self)

    def get(self, *slices, **kwds):
        """Select slices in a new representor, combine with other slices.

        :param slices: Can be various formats:

             * str:  selects slice with this name.
                     Special symbols:

                     - "*":  all slices

                     - "~": all slices with names not yet in previous ``get`` parameters are selected.

                     - "#": select first slice if all slices have a common dimension. 

             * int:   selects slice with this index

             * slice: selects slices with these indexes (note: a *Python* slice object, not an Ibidas slice)

             * representor: adds this representor slices

             * context: apply to self, adds resulting slices

             * tuple: applies get to elements in tuple, creates tuple slice from resulting slices (see .tuple() function)
             
             * list:  selected slices within list are packed using array function

        :param kwds: Same as slices, but also assigns slice names. 
        
        Examples:
            
            * str

              >>> a.get("f0","f3")

            * int
    
              >>> a.get(0, 3)

            * slice

              >>> a.get(slice(0,3))
            
            * representor

              >>> a.get(a.f0 + 3, a.f3 == "gene3",  rep(3))

            * context

              >>> a.perform_some_operation().get(_.f0 + 3, _.f3 == "gene3")
            
            * tuple

              >>> a.get((_.f0, _.f1), _.f3)

            * list

              >>> a.get([_.f0])

        """
        return repops_slice.Project(self,*slices,**kwds)

    def elements(self, name=None):
        return repops_dim.UnpackArray(self, name)

    def attributes(self, name=None):
        return repops_slice.UnpackTuple(self, name)

