import sys
import copy
import itertools
from collections import defaultdict

from query_graph import Node
from itypes import rtypes
from constants import *

_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"itypes", "dimensions","dimpaths")
_delay_import_(globals(),"wrappers","wrapper_py","wrapper")
_delay_import_(globals(),"query_context")
_delay_import_(globals(),"engines")
_delay_import_(globals(),"repops")
_delay_import_(globals(),"repops_multi")
_delay_import_(globals(),"repops_dim")
_delay_import_(globals(),"repops_slice")
_delay_import_(globals(),"repops_funcs")
_delay_import_(globals(),"ops")



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
        
    def _checkState(self,filter=RS_SLICES_KNOWN):
        if(not ((self._state & filter) == filter)):
            slices = self._getResultSlices()
            self._initialize(slices,state=RS_ALL_KNOWN | RS_INFERRED)

    def __str__(self, print_data=True):
        self._checkState(filter=RS_ALL_KNOWN)
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

        res = util.create_strtable(table)
        if(print_data):
            res += "\nData: " + str(self())
        return res

    def _getInfo(self):
        class Info(object):
            def __repr__(self):
                return self.info
        k = Info()
        k.info = self.__str__(False)
        return k
    I=property(fget=_getInfo)

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

    def _axisL(self, name):
        bname = "!L"
        while(name and (name[0] == "L" or name[0] == "R")):
            bname = bname + name[0]
            name = name[1:]

        r = repops_slice.ProjectBookmark(self,bname)
        if(name):
            r = self.__getattr__(name)
        return r
        
    def _axisR(self, name):
        bname = "!R"
        while(name and (name[0] == "L" or name[0] == "R")):
            bname = bname + name[0]
            name = name[1:]

        r = repops_slice.ProjectBookmark(self,bname)
        if(name):
            r = self.__getattr__(name)
        return r

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
        if not self._state & RS_SLICES_KNOWN:
            return []
        else:
            print ""
            print self.__str__(False)
            return [slice.name for slice in self._slices]

    def copy(self):
        res = wrapper.SourceRepresentor()
        res._initialize(tuple(self._getResultSlices(endpoint=False)))
        return res

   
    def __reduce__(self):
        return (wrapper_py.PyRepresentor, (self._getResultSlices(),))

    def __copy__(self):
        nself = Representor()
        nself.__class__ = self.__class__
        nself.__dict__ = self.__dict__.copy()
        return nself

    def getType(self):
        """Returns type of this object.
           If multiple slices, returns tuple type of slice types.
        """
        if(len(self._slices) == 1):
            return self._slices[0].type
        else:
            return rtypes.TypeTuple(False, 
                    tuple([slice.type for slice in self._slices]), 
                    tuple([slice.name for slice in self._slices]))
    Rtype=property(fget=getType)

    def getDepth(self):
        """Returns max dimension depth (number of dimensins) of
           slices in this representor. 
        """
        return max([len(slice.dims) for slice in self._slices])
    Rdepth=property(fget=getDepth)
    

    def getNames(self):
        return [slice.name for slice in self._slices]
    Rnames=property(fget=getNames)

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
        """Performs filtering on this dataset using ``condition``.
           
           :param condition: condition to filter on

                * Non-representor values are converted using ``ibidas.rep`` function

                * Representor should have single slice.

                * Can be of type bool, integer, array or slice. 

                Various data types can be used:

                * Bool: last dim should be equal to dim in this representor. Is applied to that dim by default.

                * Integer: collapses dim it is applied on. 

                * Array (of integers): selects positions indicated by integers in array.

                * Slice: selects slice from array.

           :param dim: Dim to apply the filtering on. 

                * If no dim given, applied to last common dimension of slices (except for bool types).
        """
        if(isinstance(condition, context.Context)):
            condition = context._apply(condition, self)
        return repops_multi.Filter(self, condition, dim) 

    def _getResultSlices(self, args={}, endpoint=True):
        query = query_context.QueryContext(self, args, endpoint)
        return tuple(engines.select_engine.run(query))

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
        """Cast data to new type. 

        Allowed formats:

        * single type for all slices

          >>> x.cast("int32")

        * type for each slice

          >>> x.cast(("int64","int8"))

        * type for named slices

          >>> x.cast(your_slice="int8")

        """
        return repops_slice.SliceCast(self, *newtypes, **kwds)

    def transpose(self,permute_idxs=(1,0)):
        """Transposes the dimensions of slices.
           Can only be applied to the common dimensions of slices.

           :param permute_idxs: index order of new dims.
                By default, performs matrix transpose, of first two dims, 
                i.e. permute_idxs=(1,0)

        """
        return repops_dim.PermuteDims(self,permute_idxs)

    def flat(self, dim=None,name=None):
        """Flattens (merges) a dimension with previous(parent) dim. 

        :param dim: Dim to flatten. By default, last dim.
        :param name: Name of merged dim. By default, merged name of two dimensions.
        """
        return repops_dim.Flat(self, name=name,dim=dim)

    def flatall(self, name=None):
        """Flattens all dimensions into one dimension.

        :param name: name of new merged dimension. By default, merged names of all dimensions.
        """
        return repops_dim.FlatAll(self,name=name)

    def splitDim(self,lshape,rshape,lname=None,rname=None,dimsel=None):
        """Splits dim into two dimensions.

        :param lshape: Left shape (integer or array of lengths)
        :param rshape: Right dim shape
        :param lname: New name of left dimension (default:autogenerated).
        :param rname: New name of right dimension (default:autogenerated).
        :param dimsel: Dim to split (default: last common dimension).
        """
        return repops_dim.SplitDim(self,lshape,rshape,lname,rname,dimsel)

    def harray(self):
        """Combines slices into array.
        """
        return repops_slice.HArray(self)

    def getShape(self):
        """Returns shape of all dimensions as slices in a representor object"""
        return repops_dim.Shape(self)

    def group_by(self, *args, **kwargs):
        flat = kwargs.pop("flat", {})
        group_source = self.get(*args, **kwargs)
        
        if(isinstance(flat,dict)):
            pass
        elif(isinstance(flat, list)):
            flat = {0:flat}
        else:
            flat = {0:[flat]}
            
        return repops_multi.Group(self, group_source, flat)
   
    def match(self, other, condleft, condright, group=False):
        return repops_multi.match(self, other, condleft, condright, group=group)
    
    def join(self, other, cond):
        return repops_multi.Join(self, other, cond)
    
    def match(self, other, lslice=None,rslice=None, jointype="inner"):
        return repops_multi.Match(self, other, lslice, rslice, jointype)

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
        """Rename dimensions. Similar to ``rename`` for slices.

        Shortcut: use % operator
        """
        return repops_dim.DimRename(self, *names, **kwds)
  
    def bookmark(self, *names, **kwds):
        """Bookmarks slices with a name. 
        Slices can later be accessed using attribute access, 
        with axis indicator "B". 

        Example:
            >>> x = x.bookmark("myslices")
            >>> x.Bmyslices
        """
        return repops_slice.Bookmark(self, *names, **kwds)

    def each(self, eachfunc, dtype=rtypes.unknown):
        return repops_funcs.Each(self, eachfunc=eachfunc, dtype=dtype)
    
    
    def pos(self, dim=None):
        return repops_funcs.Pos(self, dim)
    
    def argsort(self, dim=None, descend=False):
        return repops_funcs.ArgSort(self, dim, descend=descend)

    def sum(self, dim=None):
        return repops_funcs.Sum(self, dim)

    def max(self, dim=None):
        return repops_funcs.Max(self, dim)

    def min(self, dim=None):
        return repops_funcs.Min(self, dim)
    
    def argmax(self, dim=None):
        return repops_funcs.ArgMax(self, dim)

    def argmin(self, dim=None):
        return repops_funcs.ArgMin(self, dim)

    def mean(self, dim=None):
        return repops_funcs.Mean(self, dim)
    
    def median(self, dim=None):
        return repops_funcs.Median(self, dim)
      
    def any(self,dim=None):
        return repops_funcs.Any(self,dim)

    def all(self,dim=None):
        return repops_funcs.All(self,dim)

    def count(self):
        return repops_funcs.Count(self)

    def set(self, dim=None):
        return repops_funcs.Set(self,dim)
    
    def unique(self, dim=None):
        return repops_funcs.Unique(self,dim)

    def within(self, arrays):
        return repops_funcs.Within(self,arrays)

    def contains(self, elems):
        return repops_funcs.Within(elems,self)

    def array(self):
        """Packages dimension into array type"""
        return repops_dim.rarray(self)

    def tuple(self):
        """Combines slices into a tuple type"""
        return repops_slice.RTuple(self)

    def to_python(self):
        """Converts data into python data structure"""
        return repops_slice.ToPythonRep(self)()

    def sort(self, *slices, **kwargs):
        """Performs sort on data.
        
        Example:

        *  Sort slices in x on all slices. If multiple slices, combines into tuple, then sort it.

           >>> x.sort()

        *  Sort x on slice f1

           >>> x.sort(_.f1)

        * Sort x on slice f1, f3. 

          >>> x.sort(_.f1, _.f3) 
        
        For other possible sort slice selection formats, see ``get`` function. 

        """
        descend = kwargs.pop("descend",False)

        if(slices or kwargs):
            sortsource = self.get(*slices,**kwargs)
            return repops_multi.Sort(self, sortsource, descend=descend)
        else:
            return repops_multi.Sort(self, descend=descend)

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
        """Unpacks array type into dimension"""
        return repops_dim.UnpackArray(self, name)

    def attributes(self, name=None):
        """Unpacks tuple type into slices"""
        return repops_slice.UnpackTuple(self, name)

