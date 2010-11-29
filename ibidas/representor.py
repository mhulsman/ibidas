import sys
import copy
import itertools
from collections import defaultdict

_delay_import_(globals(),"utils","util","context")
_delay_import_(globals(),"itypes", "rtypes", "dimensions")
_delay_import_(globals(),"wrappers","wrapper_py")
_delay_import_(globlas(),"query_context")
_delay_import_(globals(),"engines")
_delay_import_(globals(),"repops")
_delay_import_(globals(),"repops_rel")
_delay_import_(globals(),"repops_dim")
_delay_import_(globals(),"repops_slice")
_delay_import_(globals(),"slices")

class Representor(object):
    def __init__(self, all_slices, active_slices):
        assert isinstance(all_slices, dict), "all_slices should be a dict"
        assert isinstance(active_slices, tuple), \
                                        "active_slices should be a tuple"
        assert all(slice.id in all_slices for slice in active_slices), \
                          "all active slices should occur in all_slices"

        self._all_slices = all_slices
        self._active_slices = active_slices

    def __str__(self):
        table = []
        table.append(["Slices:", "Types:", "Dims:"])
        last_dim = ()
        for slice in self._active_slices:
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
        return repops_slice.project(self, name)

    def _axisA(self, name):
        return repops_slice.unpack_tuple(self, name)

    def _axisE(self, name):
        return repops_dim.unpack_array(self, name)

    def _axisD(self, name):
        slices = self._active_dim_slice_dict[name] 
        sliceids = [slice.id for slice in slices]
        return repops_slice.ProjectId(self, sliceids)
        

    def __getattr__(self, name):
        if(not name):
            return self

        try:
            axis_letter = name[0]
            #should be upper case
            if(not axis_letter.upper() == axis_letter or axis_letter == "_"):
                #prevent recursion
                active_slices = self.__dict__["_active_slices"]
                
                #assume default
                if(len(active_slices) == 1):
                    if(name == active_slices[0].name):
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
        return [slice.name for slice in self._active_slices]

    def copy(self, modifiable=False):
        if(modifiable):
            ncls = wrapper_py.ModifiablePyRepresentor
        else:
            ncls = wrapper_py.PyRepresentor
        return ncls(self._getResult(), self._active_slices)

   
    def __reduce__(self):
        ncls = wrapper_py.PyRepresentor
        return (ncls, (self._getResult(), self._active_slices))

    def getType(self):
        if(len(self._active_slices) == 1):
            return self._active_slices[0].type
        else:
            return rtypes.TypeTuple(False, 
                    tuple([slice.type for slice in self._active_slices]), 
                    tuple([slice.name for slice in self._active_slices]))


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

    def _getResult(self, args=None):
        if(args):
            query = query_context.QueryContext(self, args)
            return engines.select_engine.run(query)
        else:
            query = query_context.QueryContext(self)
            return engines.select_engine.run(query)


    def _copyquery(self):
        return copy.copy(self)

    def __call__(self, **args):
        res = self._getResult(args)
        res = tuple([res[slice.id] for slice in self._active_slices])

        if(len(self._active_slices) == 1):
            return res[0]

        return res

    # Overloaded functions. The r* functions are implemented because you
    # want to the same behaviour no matter no which side the known object
    # resides. (a + b == b + a)
    # add ( + )
    def __add__(self, other):
        if(isinstance(other, context.Context)):
            return other.__radd__(self)
        elif(isinstance(other, repops.plusprefix)):
            pass
        return repops_rel.binop(self, other, '__add__')
    
    def __radd__(self, other):
        return repops_rel.binop(self, other, '__radd__')
    
    def __sub__(self, other):
        if(isinstance(other, context.Context)):
            return other.__radd__(self)
        elif(isinstance(other, repops.plusprefix)):
            pass
        return repops_rel.binop(self, other, '__sub__')
    
    def __rsub__(self, other):
        return repops_rel.binop(self, other, '__rsub__')

    # multiplication ( * )
    def __mul__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rmul__(self)
        elif(isinstance(other, repops.plusprefix)):
            pass
        return repops_rel.binop(self, other, '__mul__')
    
    def __rmul__(self, other):
        return repops_rel.binop(self, other, '__rmul__')
    
    # modulo ( % )
    def __mod__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rmul__(self)
        elif(isinstance(other, repops.plusprefix)):
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
        elif(isinstance(other, repops.plusprefix)):
            pass
        return repops_rel.binop(self, other, '__and__')
    
    def __rand__(self, other):
        return repops_rel.binop(self, other, '__rand__')
    
    # or operator ( | )
    def __or__(self, other):
        if(isinstance(other, context.Context)):
            return other.__ror__(self)
        elif(isinstance(other, repops.plusprefix)):
            pass
        return repops_rel.binop(self, other, '__or__')

    def __ror__(self, other):
        return repops_rel.binop(self, other, '__ror__')
    
    # exclusive-or operator ( ^ )
    def __xor__(self, other):
        if(isinstance(other, context.Context)):
            return other.__rxor__(self)
        elif(isinstance(other, repops.plusprefix)):
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
        return repops.plusprefix(self)
    
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
        return repops_slice.rtuple(self, to_python)

    def to_python(self):
        if(len(self._active_slices) > 1):
            self = self.tuple(to_python=True)

        while(self._active_slices[0].dims):
            self = self.list()

        return self()

    def sort(self, slice):
        return repops_rel.sort(self, self.get(slice))

    def _getHelperUsedNames(self, nslices, unames):
        used_names = set()
        if(not unames is None):
            used_names |= unames
        for nslice in nslices:
            for slice in nslice._active_slices:
                used_names.add(slice.name)
        return used_names 

    def get(self, *slices, **kwds):
        nslices = []
        if("_used_names_" in kwds):
            unames = kwds["_used_names_"]
            del kwds["_used_names_"]
        else:
            unames = None
            
        for elem in slices:
            if(isinstance(elem, context.Context)):
                used_names = self._getHelperUsedNames(nslices, unames)
                elem = context._apply(elem, self, get={'_used_names_':used_names})
            elif(isinstance(elem, Representor)):
                pass
            elif(isinstance(elem, tuple)):
                elem = repops_slice.rtuple(self.get(*elem))
            elif(isinstance(elem, list)):
                if(len(elem) == 1):
                    elem = self.get(*elem).array()
                else:
                    elem = self.get(*elem)
            elif(elem == "~"):
                used_names = self._getHelperUsedNames(nslices, unames)
                sliceids = [slice.id for slice in self._active_slices if slice.name not in used_names]
                elem = repops_slice.ProjectId(self, sliceids)
            elif(elem == "#"):
                common_dims = set([slice.dims for slice in self._active_slices])
                if len(common_dims) != 1:
                    raise RuntimeError, "Cannot use # selector as fields do not have a common dimension"
                if(unames):
                    elem = repops_slice.project(self, iter(unames).next())                    
                else:
                    elem = repops_slice.project(self, 0)
            else:
                elem = repops_slice.project(self, elem)
            nslices.append(elem)

        for name, elem in kwds.iteritems():
            if(isinstance(elem, context.Context)):
                used_names = self._getHelperUsedNames(nslices, unames)
                elem = context._apply(elem, self, _used_names_=used_names)
            elif(isinstance(elem, Representor)):
                pass
            elif(isinstance(elem, tuple)):
                elem = repops_slice.rtuple(self.get(*elem))
            elif(isinstance(elem, list)):
                if(len(elem) == 1):
                    elem = self.get(*elem).array()
                else:
                    elem = self.get(*elem)
            else:
                elem = repops_slice.project(self, elem)
            nslices.append(elem/name)
       
        if(len(nslices) > 1):
           return repops_slice.combine(*nslices)
        else:
            assert nslices, "No slice selected with get"
            return nslices[0]

    def _getActiveDimDict(self):
        if(not hasattr(self, '_active_dim_cache')):
            res = defaultdict(set)
            for slice in self._active_slices:
                for dim in slice.dims:
                    res[dim.name].add(dim)
            self._active_dim_cache = res
        return self._active_dim_cache
    _active_dim_dict = property(_getActiveDimDict)

    def _getActiveSliceDict(self):
        if(not hasattr(self, '_active_slice_cache')):
            res = defaultdict(set)
            for slice in self._active_slices:
                res[slice.name].add(slice)
            self._active_slice_cache = res
        return self._active_slice_cache
    _active_slice_dict = property(_getActiveSliceDict)

    def _getActiveDimSliceDict(self):
        if(not hasattr(self, '_active_dim_slice_cache')):
            res = defaultdict(set)
            for slice in self._active_slices:
                for dim in slice.dims:
                    res[dim.name].add(slice)
            self._active_dim_slice_cache = res
        return self._active_dim_slice_cache
    
    _active_dim_slice_dict = property(_getActiveDimSliceDict)


    def _getActiveDimParentDict(self):
        if(not hasattr(self, '_active_dim_parent_cache')):
            res = defaultdict(set)
            for slice in self._active_slices:
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
            for slice in self._active_slices:
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
            for slice in self._active_slices:
                for dim in slice.dims:
                    res[dim.id] = dim
            self._active_dim_id_cache = res
        return self._active_dim_id_cache
    _active_dim_id_dict = property(_getActiveDimIdDict)

    def _getActiveSliceIdDict(self):
        if(not hasattr(self, '_active_slice_id_cache')):
            res = {}
            for slice in self._active_slices:
                res[slice.id] = slice
            self._active_slice_cache = res
        return self._active_slice_id_cache
    _active_slice_id_dict = property(_getActiveSliceIdDict)

    def _getActiveDimIdSliceDict(self):
        if(not hasattr(self, '_active_dim_id_slice_cache')):
            res = defaultdict(set)
            for slice in self._active_slices:
                for dim in slice.dims:
                    res[dim.id].add(slice)
            self._active_dim_id_slice_cache = res
        return self._active_dim_id_slice_cache
    _active_dim_id_slice_dict = property(_getActiveDimIdSliceDict)


    def _getActiveDimIdParentDict(self):
        if(not hasattr(self, '_active_dim_id_parent_cache')):
            res = defaultdict(set)
            for slice in self._active_slices:
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
            for slice in self._active_slices:
                for pdim, cdim in zip(slice.dims, slice.dims[1:]):
                    res[pdim.id].add(cdim)
                if(slice.dims):
                    res[slice.dims[-1].id].add(None)
                    res[None].add(slice.dims[0])
            self._active_dim_id_child_cache = res
        return self._active_dim_id_child_cache
    _active_dim_id_child_dict = property(_getActiveDimIdChildDict)       

    def _createSlicePaths(self, dim_selector=None):
        """Creates slice paths: dict with slice id to slice dims.
           Parameters
           ----------
           dim_selector: (optional) Given to determine which slices
               to take. Paths are shortened up to dims indicated by dim_selector.
               Possibilities (see _identifyDim)
                  - tuple of dim namees / dims
                  - index, dim name, slice name, dim, slice

           Returns
           -------
           slicepaths: dict with slice id to slice path (tuple of dims)
           path_dims: if dim selector is given, returns dims covered by dims
                        indicated by dim_selector
           actual_dims: returns list of dims actually indicated by dim selector
        """
        
        if(dim_selector is None):
            slicepaths = dict([(slice.id, slice.dims) for slice in self._active_slices])
            return (slicepaths, (), ())
        
        path = self._identifyDim(dim_selector)
        if(isinstance(path, bool)):
            return path
        
        if(isinstance(dim_selector, tuple)):
            #multiple dims, match all
            pos = 0
            path_dims = []
            actual_dims = []
            for dim in path:
                if(dim.name == dim_selector[pos] or dim == dim_selector[pos]):
                    path_dims.append(dim)
                    pos += 1
                    actual_dims.append(dim)
                    if(len(dim_selector) == pos):
                        break
                else:
                    if(pos > 0):
                        path_dims.append(dim)
            for slice in self._active_slices:
               res = util.contained_in(path, slice.dims) 
               if(not res is False):
                   firstpos = slice.dims.index(actual_dims[0])
                   slicepaths[slice.id] = slice.dims[:firstpos]
            path_dims = tuple(path_dims)
            actual_dims = tuple(actual_dims)
        else:
            #no tuple, only match last dim from dim identifier
            #only last part of path to first fixed dim
            for pos in range(len(path) - 1):
                if(not path[-(pos + 1)].variable):
                    path = path[-(pos + 1):]
                    break
            for slice in self._active_slices:
               res = util.contained_in(path, slice.dims) 
               if(not res is False):
                   slicepaths[slice.id] = slice.dims[:(res[1]-1)]
            path_dims = (path[-1],)
            actual_dims = path_dims
        return (slicepaths, path_dims, actual_dims)


    def _matchSlicePaths(self, dimpath, slicepaths):
        """Matches dimpath (tuple of dims) uniquely to paths in slicepaths or fails.
        
        Parameters
        ----------
        dimpath: tuple of dims
        slicepaths: dictionary of slice id to slice path

        Dims in dimpath that also occur in slice paths are matched to these dims,
        in order of occurence in dimpath. Dims that are in dimpath, but not in 
        the slice paths are matched as late as possible. 
        If dim path is longer than matcheable path, last dims in path and same_dim 
        will contain None values. 


        Returns
        -------
        False if cannot match (e.g. no unique match possible)
        path: path covered by dims in dimpath
        same_dim: dims matched, which are also in dimpath
        """

        curpos = dict([(id, 0) for id in slicepaths.keys()])
        result = []
        same_dim = []
        if not self._matchSlicePathsHelper(dimpath, slicepaths,  result, same_dim, curpos):
            return False
        result = result[::-1]
        same_dim = same_dim[::-1]
        return (result, same_dim)


    def _matchSlicePathsHelper(self, dimpath, slicepaths, result, same_dim, curpos):
        """Matches dimpath with paths in slicepaths, where path begin pos is 
        stored in curpos.  Resulting matched path is stored in result,
        new dimensions for dimpath are stored in same_dim. 
        """
        if not dimpath:
            return True

        ncurpos = {}
        dim = dimpath[0]
        for id, pos in curpos.iteritems():
            path = slicepaths[id]
            if(dim in path[pos:]):
                pos = path.index(dim)
                curpos[id] = pos
                ncurpos[id] = pos + 1

        nextpath = dimpath[1:]
        if(ncurpos):
            if(not nextpath):
                result.append(dim)
            else:
                #remove unmatched slices from curpos
                for id in curpos.keys():
                    if(not id in ncurpos):
                        del curpos[id]
                #match next one
                if not self._matchSlicePathsHelper(nextpath, slicepaths, 
                                                   result, same_dim, ncurpos):
                    return False
                #extend result path if common
                rpaths = set()
                for id, pos in ncurpos.iteritems():
                    rpaths.add(slicepaths[id][curpos[id]:pos])
                if(len(rpaths) != 1):
                    return False
                result.extend(rpaths.pop()[::-1])
            same_dim.append(dim)
        else:
            if(not nextpath):
                rpaths = set([slicepaths[id][pos:] for id, pos in curpos.iteritems()])
                rpaths.discard(())
                if(len(rpaths) == 0 and not any([not elem is None for elem in result])):
                    rdim = None
                elif(len(rpaths) != 1):
                    return False
                else:
                    rpath = rpaths.pop()
                    rdim = rpath[-1]
                    for id, pos in curpos.iteritems():
                        curpos[id] = pos + len(rpath)
            else:
                #keep some place for myself
                for id, pos in curpos.iteritems():
                    ncurpos[id] = pos + 1
                #match next one
                if not self._matchSlicePathsHelper(nextpath, slicepaths, 
                                                   result, same_dim, ncurpos):
                    return False
                #match myself to last non-matched dim in remaining slices
                #shoudl be unique dim
                rdims = set()
                for id, pos in ncurpos.iteritems():
                    rdims.add(slicepaths[id][pos-1])
                    curpos[id] = pos -1
                if(len(rdim) != 1):
                    return False
                rdim = rdim.pop()
            result.append(rdim)
            same_dim.append(rdim)
            
        return True

    def _commonDims(self):
        """Returns common dimensions shared by all slices"""
        pos = 0
        maxlen = min([len(slice.dims) for slice in self._active_slices])
        if(len(self._active_slices) == 0):
            return ()
        
        while(pos < maxlen):
            cd = set([slice.dims[pos] for slice in self._active_slices])
            pos += 1
            if(len(cd) != 1):
                break
        
        return self._active_slices[0].dims[:pos]

    def _identifyDim(self, dim_selector=None):
        """Identifies a dim, using dim_selector. 
        Returns dim identifier, which is a tuple of dims.
            If fixed dim identified:
                only fixed dim
            If variable dim, path: 
                tuple with fixed dim --> --> variable dim
            However, if a path is specified (tuple of dims,
               slice dims, unique index path dims), then 
               path is always contained in tuple
               even if a fixed dim in it is not at the beginning.
            multiple matching dims: True
            no matching dims: False

        
        Parameters
        ----------
        dim_selector: 
            if None: 
                return unique common dimension if exists
            if string:
                search for dim with name.
                    if only one fixed dim: return
                    if variable dim with unique path: return
                    otherwise returns None
                search for slice with name 
                    return identifier for last dim
            if dim:
                if only one fixed dim: return
                if variable dim with unique path: return
                otherwise returns None
            if slice
                return identifier for last dim
            if tuple of dim (names):
                if empty: same as None
                searches for dim name, identified by branch in tuple. 
                Starts from last dim, find unique path to fixed dim.
                If exist, returns, otherwise, returns None
            if int:
                if there is a unique dim path, use that to index.
                If positive, dim paths can have variable length as 
                long as they are equal on the base
        """
        if(isinstance(dim_selector, tuple) and len(dim_selector) == 0):
            dim_selector = None

        if(dim_selector is None):
            res =  self._commonDims()
            return res
        elif(isinstance(dim_selector, int)):
            fdims = set([slice.dims for slice in self._active_slices 
                                    if slice.dims])
            if(len(fdims) == 1):
                path = fdims.pop()
            else:
                path = []
                for elems in itertools.izip_longest(*list(fdims)):
                    dimset = set([elem for elem in elems if not elem is None])
                    if(len(dimset) == 1):
                        path.append(dimset.pop())
                    else:
                        return True

                path = tuple(path)
            
            if(not path):
                return False
            if(dim_selector == -1):
                return self._identifyDimPathHelper(path)
            else:
                return self._identifyDimPathHelper(path[:(dim_selector + 1)])
            

        elif(isinstance(dim_selector, str)):
            if(dim_selector in self._active_dim_dict):
                nselectors = self._active_dim_dict[dim_selector]
            elif(dim_selector in self._active_slice_dict):
                nselectors = self._active_slice_dict[dim_selector]
            else:
                return False

            results = []
            for selector in nselectors:
                res = self._identifyDim(selector)
                if(not res is False):
                    results.append(res)
            if(len(results) == 1):
                return results[0]
            elif(len(results) == 0):
                return False
            else:
                return True
        elif(isinstance(dim_selector, dimensions.Dim)):
            return self._identifyDimPathHelper((dim_selector,))
        elif(isinstance(dim_selector, slices.Slice)):
            return self._identifyDimPathHelper(dim_selector.dims)
        elif(isinstance(dim_selector, tuple)):
            return self._identifyDimPathHelper(dim_selector)
        else:
            raise RuntimeError, "Unexpected dim selector: " + str(dim_selector)

    def _identifyDimPathHelper(self, path):
        """Given a path (tuple of dims or dim names), 
        Determines if there is a matching dim path in self that uniquely 
        identifies the dimension last in path. Such a path should have the 
        same order of dimenmsions as given in 'path', but not necessarily the
        same dimensions. It however should be unique.
        
        Returns an identifier for the last dimension if unique path found.
        (See _identifyDim for description).

        Note: paths that have a fixed dimension which is not on the beginning
        will be shortened such that they start from this fixed dimension. The 
        beginning of the path will not be matched in that case!

        Returns
        -------
        True: if multiple paths are possible
        False: if no path is matching
        dimension identifier otherwise
        """
        if(not path):
            return False
        reslist = []
        active_dim_dict = self._active_dim_dict
        active_dim_id_dict = self._active_dim_id_dict
        parent_dim_id_dict = self._active_dim_id_parent_dict

        for pos, dim in enumerate(path[::-1]):
            if(isinstance(dim, dimensions.Dim)):
                if not dim.id in active_dim_id_dict:
                    return False
            else:
                dims = active_dim_dict[dim]
                if(len(dims) > 1): #multiple matches for dim name. Try them all.
                    sresults = []
                    for dim in dims:
                        sres = self._identifyDimPathHelper(path[:(-(pos + 1))] + (dim,))
                        if(not sres is False):
                            sresults.append(sres)
                    if(len(sresults) > 1):
                        return True
                    elif(len(sresults) == 0):
                        return False
                    reslist.extend(sresults[0][::-1])
                    break
                elif(len(dims) == 1):
                    dim = iter(dims).next()
                else:
                    return False
                   
            if(not reslist): 
                reslist.append(dim)
                match = True
            else:
                match = False
            
            while(match is False):
                parentdims = parent_dim_id_dict[reslist[-1].id]
                if(dim in parentdims):
                    reslist.append(dim)
                    break
                
                if(len(parentdims) > 1):
                    sresults = []
                    for pdim in parentdims:
                        sres = self._identifyDimPathHelper(
                                        path[:(-(pos + 1))] + (dim, pdim))
                        if(not sres is False):
                            sresults.append(sres)
                    if(len(sresults) > 1):
                        return True
                    elif(len(sresults) == 0):
                        return False
                    reslist.extend(sresults[0][::-1])
                    break
                elif(len(parentdims) == 1):
                    reslist.append(iter(parentdims).next())
                else:
                    return False
        else:
            while(True):
                parentdims = parent_dim_id_dict[reslist[-1].id]
                if(len(parentdims) > 1):
                    return True
                elif(len(parentdims) == 0):
                    break
                else:
                    nelem = iter(parentdims).next()
                    if(nelem is None):
                        break
                    reslist.append(nelem)
        
        return tuple(reslist[::-1])


