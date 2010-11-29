from collections import defaultdict
from itertools import chain
import numpy
import math

from repops import *
from repops_dim import *
from repops_slice import *

_delay_import_(globals(),"utils","util","context","cutils")
_delay_import_(globals(),"itypes","dimensions","rtypes")
_delay_import_(globals(),"slices")

class RFilter(MultiOpRep):#{{{
    """Filter data based on constraint"""
    def __init__(self, source, constraint, dim=False):
        """
        Parameters
        ----------
        source: source to be filtered
        constraint: constraint to be used to determine which 
                    elements to select. Can be representor of bool type,
                    index, slice, array of bools, array of ints.
        dim: (Optional) dim to filter. To override auto determination.

        Filter chooses which dim to filter based on the dimension of constraint
        if it is a representor. If not, it will determine if their is a unique
        dimension branch, and will select the last dimension from it. 
        """
       
        #create set of constraint dims
        constraint_slices = []
        if(not isinstance(constraint, representor.Representor)):
            constraint = wrapper_py.rep(constraint)

        assert len(constraint._active_slices) == 1, \
            "Filter constraint should have only 1 slice."
        cslice = constraint._active_slices[0]        
        constraint_dims = cslice.dims

        if(isinstance(cslice.type, rtypes.TypeBool) or cslice.type.has_missing):
            #if not a bool, make it a bool by performing ~ismissing on the constraint
            if(not isinstance(cslice.type, rtypes.TypeBool)):
                constraint = ~ismissing(constraint)
            
            #when bool data is given, filtered dimenson is specified by the constraint
            assert dim is False, "Cannot use bool or missing data type and also \
                               specify filter dimension. Dimension of constraint \
                               determines filter dimension."

            #determine prefix and suffix dimensions from constraint (i.e. suffix dims are
            #those dimensions that can stand alone from earlier dimensions).
            path_prefix, path_suffix = dim_helpers.pathSuffix(constraint_dims)
            
            #now determine filterdim path by matchin prefix and suffix to 
            #active slices (suffix has to match exactly, prefix can be broadcast)
            filter_dimpath = dim_helpers.broadcastAndMatchDimPath(source._active_slices, path_prefix, path_suffix)

            #create broadcast plan for constraint path againt filter path
            broadcast_plan, new_dimpath = dim_helpers.planBroadcast(filter_dimpath[:-1], constraint_dims[:-1])
           
            #adapt dimensions of to be filtered slices
            ndim = dimensions.Dim(UNDEFINED, 
                                  len(constraint_dims) - 1, filter_dimpath[-1].has_missing,
                                  name = "f" + filter_dimpath[-1].name
                                  )
            new_dimpath += (ndim,)
            
            #determine new dimensions of constraint dimensions with respect to end of path
            #--> used for adapting possible variable dimensions
            varadapt = (0,) + dim_helpers.planPattern(broadcast_plan[::-1], 1)

        else:
            #not boolean, but position based filtering

            #if no dim given, use last dim in unique dim path
            if(dim is False):
                dim = -1
            
            #determine dim path to filter on 
            path_suffix = dim_helpers.identifyDimPath(source, dim)
            assert isinstance(path_suffix, tuple), "Cannot find unique matching " + \
               "dimension path for constraint."

            
            #if array or slice type
            if(isinstance(cslice.type, (rtypes.TypeArray, rtypes.TypeSlice))):
                #if array type, determine if it contains integers
                if(isinstance(cslice.type, rtypes.TypeArray)):
                    assert isinstance(cslice.type.subtypes[0], rtypes.TypeInteger) and \
                            not isinstance(cslice.type.subtypes[0], rtypes.TypeBool), \
                            "Subtype of an array should be integer (excluding bool)"
                    #unpack
                    constraint = constraint.E
               
                #last dimension is matched by array or slice
               
                #now determine filterdim path by matchin prefix and suffix to 
                #active slices (suffix has to match exactly, constraint_dims can be broadcast)
                filter_dimpath = dim_helpers.broadcastAndMatchDimPath(source._active_slices, constraint_dims, path_suffix, single_match_suffix=True)
                
                #determine broadcast plan and new dim path
                broadcast_plan, new_dimpath = dim_helpers.planBroadcast(filter_dimpath[:-1], constraint_dims)
                
                #adapt dimensions of to be filtered slices
                ndim = dimensions.Dim(UNDEFINED, 
                                    len(constraint_dims), filter_dimpath[-1].has_missing,
                                    name = "f" + filter_dimpath[-1].name
                                    )
                new_dimpath += (ndim,)
                
                #determine new dimensions of constraint dimensions with respect to end of path
                #--> used for adapting possible variable dimensions
                varadapt = (0,) + dim_helpers.planPattern(broadcast_plan[::-1], 1)

            elif(isinstance(cslice.type, rtypes.TypeInteger)):
                if(not constraint_dims):
                    broadcast_plan, new_dimpath = dim_helpers.planBroadcast(path_suffix[:-1], constraint_dims)
                    
                    varadapt = (-1,) + dim_helpers.planPattern(broadcast_plan[::-1], 1, startval=-1)
                    filter_dimpath = path_suffix                   
                else:
                    filter_dimpath = dim_helpers.broadcastAndMatchDimPath(source._active_slices, constraint_dims, path_suffix, single_match_suffix=True)

                    #last dim has matched before filter dim, split to integer
                    if(constraint_dims[-1] in filter_dimpath[:-1]):
                        broadcast_plan, new_dimpath = dim_helpers.planBroadcast(filter_dimpath[:-1], constraint_dims)
                        
                        varadapt = (-1,) + dim_helpers.planPattern(broadcast_plan[::-1], 1, startval=-1)
                    else:
                        broadcast_plan, new_dimpath = dim_helpers.planBroadcast(filter_dimpath[:-1], constraint_dims[:-1])
                        
                        #adapt dimensions of to be filtered slices
                        ndim = dimensions.Dim(UNDEFINED, 
                                            len(constraint_dims) - 1, filter_dimpath[-1].has_missing,
                                            name = "f" + filter_dimpath[-1].name
                                            )
                        new_dimpath += (ndim,)
                        
                        varadapt = (0,) + dim_helpers.planPattern(broadcast_plan[::-1], 1)
            else:
                raise RuntimeError, "Cannot use " + str(cslice.type) + " as filter."

        #now use filter path to find slices that should be filtered
        match_slices, start_depths = dim_helpers.matchDimPath(source._all_slices.values(), filter_dimpath)
        nslices, startdepths = dim_helpers.redimMatch(match_slices, start_depths, filter_dimpath, new_dimpath, varadapt)

        nall_slices = source._all_slices.copy()
        for slice in nslices:
            nall_slices[slice.id] = slice

        nactive_slices = [nall_slices[slice.id] for slice in source._active_slices]

        cslice = constraint._active_slices[0]        
        self._constraint_slices = [cslice]
        self._filter_slices = [nslices]
        self._filter_depths = [start_depths]
        self._filter_broadcast = [broadcast_plan]
        MultiOpRep.__init__(self, (source, constraint), nall_slices, tuple(nactive_slices))
        return
       
#}}}
rfilter = delayable()(RFilter)

class ifilter(MultiOpRep):#{{{
    """Filter data based on constraint"""
    def __init__(self, source, constraint, dim=None):
        """
        Parameters
        ----------
        source: source to be filtered
        constraint: constraint to be used to determine which 
                    elements to select. Can be representor of bool type,
                    index, slice, array of bools, array of ints.
        dim: (Optional) dim to filter. To override auto determination.

        Filter chooses which dim to filter based on the dimension of constraint
        if it is a representor. If not, it will determine if their is a unique
        dimension branch, and will select the last dimension from it. 
        """
        
        
        #create set of constraint dims
        constraint_slices = []
        if(not isinstance(constraint, representor.Representor)):
            constraint = wrapper_py.rep(constraint)
            cdimbranch = source._identifyDim(dim)
            assert not isinstance(cdimbranch, bool), \
                                "Could not find (unique) dimension to filter"
            cslice = constraint._active_slices[0]
        else:
            assert len(constraint._active_slices) == 1, \
                "Filter constraint should have only 1 slice."
            
            cslice = constraint._active_slices[0]
            if(not isinstance(cslice.type, rtypes.TypeBool) and cslice.type.has_missing):
                constraint = ~ismissing(constraint)
                cslice = constraint._active_slices[0]
            assert isinstance(cslice.type, rtypes.TypeInteger), \
                "Filter constraint should be of bool/integer/has_missing type."
            cdimbranch = cslice.dims
            
            dimdict = source._active_dim_id_dict
            while(cdimbranch and not cdimbranch[-1].id in dimdict):
                if isinstance(cslice.type, rtypes.TypeBool):
                    constraint_slices.append(cslice)
                    cslice = slices.saggregate(cslice, exec_any, check_bool)
                    cdimbranch = cslice.dims
                else:
                    cdimbranch = cdimbranch[:-1]

            if(not cdimbranch):
                assert len(cslice.dims) <= 1, \
                                "Could not match nested index to data"

                cdimbranch = source._identifyDim(dim)
                assert cdimbranch, "Could not find (unique) dimension to filter"
            elif(not isinstance(cslice.type, rtypes.TypeBool)):
                if(len(cdimbranch) != len(cslice.dims)):
                    assert len(cdimbranch) == len(cslice.dims) - 1, \
                        "Dims of constraint do not match dims of filter source."
                
                    active_dim_id_child = source._active_dim_id_child_dict
                    childdims = active_dim_id_child[cdimbranch[-1].id]
            
                    assert len(childdims) <= 1, "Cannot find unique " + \
                        "dimension to apply filter to: " + str(childdims)

                    assert len(childdims) > 0, "Cannot find dimension " + \
                                                       "to apply filter to."

                
                    cdimbranch = cdimbranch + (iter(childdims).next(),)

        constraint_slices.append(cslice)
                

        oldbranch = cdimbranch
        newbranch = list(cdimbranch)
        
        odim = oldbranch[-1]
        if(not cslice.dims and isinstance(cslice.type, rtypes.TypeInteger)):
            newbranch.pop()
        else:
            ndim = dimensions.Dim(UNDEFINED, 
                                  odim.variable, odim.has_missing)
            newbranch[-1] = ndim
        newbranch = tuple(newbranch)
        nall_slices, filter_slices = redimSlices3(source._all_slices, 
                                        oldbranch, newbranch, return_redim=True)
        for slice in constraint_slices:
            nall_slices[slice.id] = slice
        
        nactive_slices = [nall_slices[slice.id] 
                                    for slice in source._active_slices]

        self._constraint_slices = [cslice]
        self._filter_slices = [filter_slices]
        self._filter_depth = [len(oldbranch)-1]
        self._filter_branch = oldbranch
        
        MultiOpRep.__init__(self, (source, constraint), nall_slices, 
             tuple(nactive_slices))#}}}

class join_filter(RFilter):
    pass

class Join(MultiOpRep):
    
    def _axisL(self, name):
        active_slice_ids = set([slice.id for slice in 
                    self._sources[0]._active_slices])
        res = ProjectId(self, active_slice_ids)
        if(name):
            res = getattr(res, name)
        return res

    def _axisR(self, name):
        active_slice_ids = set([slice.id for slice in 
                                self._sources[1]._active_slices])
        res = ProjectId(self, active_slice_ids)
        if(name):
            res = getattr(res, name)
        return res

def _join(lsource, rsource, condition=None, ldim=(), rdim=(), 
         type="INNER", group=False):
    """
    Parameters
    ----------
    lsource, rsource: the two sources to be joined
    condition: condition to determine which dimension-element pairs
               are kept.
    ldim, rdim: override automatical determination on which dims to join.
    type: "INNER", "LEFT", "RIGHT", "OUTER"
    group: nest the right source instead of expanding the join dimension.

    
    JOINING nested data is a bit more complicated than joining two 
    one-dimensional tables. When joining, one combines two dimensions
    making them one. In nested data there are multiple dimensions, and thus
    multiple possible choices. 
    
    Rules:
    - In the left source, each dimension can be used as join dimension
    - In the right source, the dim has to be always a 'top' dimension. 
      e.g. one should think of this as joining two paths. In the right 
      source one has always to start with a top dimension, in the left source
      one can use various 'bottom' (and top) dimensions. 
    
    Determining the join-dimensions
     1) if ldim / rdim specified use them to identify dim branch
     2) if left only 1 dimension, use that one.
     3) if right only 1 top dimension, use that one
     4) If a condition is given, we use that to determine the dimensions
     5) From the left source, if there is only 1 top dimension, 
       we choose that one 
     otherwise: error
    """
    
    #rule 1-3
    if(not ldim is None):
        ldim = lsource._identifyDim(ldim) 

    if(not rdim is None):
        rdim = rsource._identifyDim(rdim)

    #rule 4
    if(not condition is None):
        assert isinstance(condition, context.Context), \
            "Condition should be a context object (use context operator _)"
        if(ldim is None or rdim is None):
            #still to be implemented
            pass

    #rule 5
    if(not isinstance(ldim, tuple)):
        ldim = lsource._identifyDim()
    if(not isinstance(rdim, tuple)):
        rdim = rsource._identifyDim()

    assert isinstance(ldim, tuple), "Join dimension of left source could not be determined." + \
            "Please specify using ldim='name of dim' in join command."

    assert isinstance(rdim, tuple), "Join dimension of right source could not be determined." + \
            "Please specify using rdim='name of dim' in join command."
   

    #determine new dims
    oleftbranch = ldim
    orightbranch = rdim
    
    #get right slice ids
    rdimslices = matchDimSlices(rsource._all_slices, orightbranch)
    rsliceids = set(rdimslices.keys())
  
    if(not group):
        bcdims, mouter, minner, nmouter, nminner =  broadcastDims(oleftbranch[:-1], orightbranch[:-1])
        assert not nmouter and not nminner, "Multiple right dimensions, but do not match left dimensions"

        
        join_dim = dimensions.Dim(UNDEFINED, 
                                variable=oleftbranch[-1].variable or orightbranch[-1].variable, 
                                        has_missing=oleftbranch[-1].has_missing or orightbranch[-1].has_missing)
        leftbranch = list(ldim)
        leftbranch[-1] = join_dim
        leftbranch = tuple(leftbranch)
        rightbranch = leftbranch


        lall_slices, lslices = redimSlices3(lsource._all_slices, 
                                    oleftbranch, leftbranch, return_redim=True)
        if(type == "LEFT" or type == "OUTER"):
            pass
            #add has_missing to type on right side
            #add has_missing to child dims on right side

        if(type == "RIGHT" or type == "OUTER"):
            pass
            #add has_missing to type on left side
            #add has_missing to child dims on left side
   
    else:
        bcdims, mouter, minner, nmouter, nminner =  broadcastDims(oleftbranch, orightbranch[:-1])
        assert not nmouter and not nminner, "Multiple right dimensions, but do not match left dimensions"

        leftbranch = oleftbranch
        rightbranch = oleftbranch + nminner + orightbranch[-1:]
        
        lall_slices = lsource._all_slices
        ldimslices = matchDimSlices(lsource._all_slices, oleftbranch)
        lslices = set(ldimslices.values())

    #realias slices that are similar between the joins
    #and take part in the join
    rsource = realiasSimilarSlices(rsource, lall_slices, 
                            keep_equal=True, always_disimilar_ids=rsliceids)
    rall_slices, rslices = redimSlices3(rsource._all_slices, 
                                orightbranch, rightbranch, return_redim=True)
       
   
    nall_slices = lall_slices.copy()
    nall_slices.update(rall_slices)
    nactive_slices = tuple([nall_slices[slice.id] for slice in 
                    chain(lsource._active_slices, rsource._active_slices)])

    self = Join((lsource, rsource), nall_slices, nactive_slices)

    if(not group):
        lsliceids = set([slice.id for slice in lslices])
    else:
        lsliceids = set([iter(lslices).next().id])
    rsliceids = set([slice.id for slice in rslices])

    self._join_slices = (lsliceids, rsliceids)
    self._join_branches = (tuple(leftbranch), tuple(rightbranch))
    self._join_broadcast = bcdims
    self._join_group = group
    
    return self


def join(lsource, rsource, condition=None, ldim=None, rdim=None,
         type="INNER", group=False):
    self = _join(lsource, rsource, condition, ldim, rdim, type, group)
    
    if(condition):
        condition = context._apply(condition, self)
        res = join_filter(self, condition)
        assert len(res._filter_broadcast) == len(self._join_broadcast), \
            "Condition does not filter at join level"
        #ssert res._filter_branch == self._join_branches[0], \
        #   "Condition does not filter at join dimensions"

        return res
    else:
        return self

class EquiJoin(Join):
    pass

def match(lsource, rsource, left=None, right=None, 
         type="INNER", group=False, both=None, ldim=None, rdim=None):
    if(both):
        left = right = both
    if(ldim is None):
        condl = lsource.get(left)
        ldim = condl._active_slices[0].dims

    if(rdim is None):
        condr = rsource.get(right)
        rdim = condr._active_slices[0].dims
   
    self = _join(lsource, rsource, None, ldim, rdim, type, group)
    
    sleft = self.L
    sright = self.R
    if(left in sleft._active_slice_dict and right in sright._active_slice_dict):
        lsliceset = sleft._active_slice_dict[left]
        rsliceset = sright._active_slice_dict[right]
        assert len(lsliceset) == 1, "Cannot find " + left + " in left source"
        assert len(rsliceset) == 1, "Cannot find " + right + " in right source"
        lslice = iter(lsliceset).next()
        rslice = iter(rsliceset).next()

        self._lsliceid = lslice.id
        self._rsliceid = rslice.id
        self.__class__ = EquiJoin
        if(left == right and not group):
            self._active_slices = util.delete_from_tuple(self._active_slices, lsliceset)
        return self
    else:
        condl = sleft.get(left)
        condr = sright.get(right)
        res = join_filter(self, condl == condr)
    
        assert len(res._filter_broadcast) == len(self._join_broadcast), \
            "Condition does not filter at join level"
        #assert res._filter_branch == self._join_branches[0], \
        #    "Condition does not filter at join dimensions"
        return res

@delayable()
def sort(source, sortslice):
    constraint = argsort(sortslice)
    return rfilter(source, constraint)

@delayable()
def rmap(source, exec_func, otype=rtypes.unknown, dim=None, *params, **kwds):#{{{
    if(isinstance(otype, str)):
        otype = rtypes.createType(otype)
    type_func = lambda x, y : otype
    
    return apply_slice(source, "map", slices.MapSlice, dim, exec_func, type_func)#}}}
    


def create_mapseq(name, exec_func, type_func=None, otype=None, defslice="*"):#{{{
    if(type_func is None and otype is None):
        type_func = lambda x, y : rtypes.unknown
    elif(type_func is None):
        if(isinstance(otype, str)):
            otype = rtypes.createType(otype)
        type_func = lambda x, y : otype

    @delayable(defslice) 
    def apply_func(source, *params, **kwds):
        dim = kwds.get("dim", None)
        return apply_slice(source, name, slices.MapSeqSlice, dim, exec_func, type_func, *params, **kwds)
    
    return apply_func#}}}


def exec_pos(seq):
    return numpy.arange(len(seq), dtype="uint32").view(util.farray)

def exec_bin(seq, nbin=10):
    res = int(math.ceil(len(seq) / float(nbin)))
    repeat_count = numpy.zeros(nbin, dtype="uint32")
    repeat_count[:] = res
    rest = ((res * nbin) - len(seq))
    if(rest):
        repeat_count[-rest:] -= 1
    return numpy.repeat(numpy.arange(nbin, dtype="uint32"), list(repeat_count))

def exec_argsort(seq):
    return numpy.argsort(seq).view(util.farray)

def exec_ismissing(seq):
    if(issubclass(seq.dtype.type, float)):
        return numpy.isnan(seq).view(util.farray)
    else:
        mtype = rtypes.MissingType
        
        return cutils.darray([elem.__class__ is mtype 
                    for elem in seq], bool).view(util.farray)




pos = create_mapseq("pos", exec_pos, otype="uint32", defslice="#")
rbin = create_mapseq("bin", exec_bin, otype="uint32", defslice="#")
argsort = create_mapseq("argsort", exec_argsort, otype="int64")
ismissing = create_mapseq("ismissing", exec_ismissing, otype="bool")

def create_aggregate(name, exec_func, type_func=None, otype=None, defslice="*"):#{{{
    if(type_func is None and otype is None):
        type_func = lambda x, y, z: rtypes.unknown
    elif(type_func is None):
        if(isinstance(otype, str)):
            otype = rtypes.createType(otype)
        type_func = lambda x, y, z: otype

    @delayable(defslice)
    def apply_func(source, dim=None):
        return apply_slice(source, name, slices.AggregateSlice, dim, exec_func, type_func)
    return apply_func#}}}



def exec_any(seq):
    return seq.any()

def exec_all(seq):
    return seq.all()

def exec_sum(seq):
    return seq.sum()

def exec_mean(seq):
    return seq.mean()

def exec_max(seq):
    return seq.max()

def exec_min(seq):
    return seq.min()

def exec_count(seq):
    return len(seq)

def check_arith(in_type, dim, exec_func):
    assert isinstance(in_type, rtypes.TypeNumber), \
        "Performing arithmetic aggregration on non-number type"
    dv = in_type.toDefval()
    dv = numpy.array([dv])
    ntype = rtypes.createType(exec_func(dv).dtype)
    if(in_type.has_missing):
        ntype.has_missing = True
    return ntype

def check_bool(in_type, dim, exec_func):
    return rtypes.TypeBool()

rany = create_aggregate("any", exec_any, otype="bool")
rall = create_aggregate("all", exec_all, otype="bool")
rsum = create_aggregate("sum", exec_sum, type_func=check_arith)
mean = create_aggregate("mean", exec_mean, type_func=check_arith)
rmax = create_aggregate("max", exec_max, type_func=check_arith)
rmin = create_aggregate("min", exec_min, type_func=check_arith)
count = create_aggregate("count", exec_count, otype="uint32", defslice="#")

def exec_set(seq):
    return frozenset(seq)

def check_set(in_type, dim, exec_func):
    ndim = dimensions.Dim(UNDEFINED, 
                          dim.variable, dim.has_missing)
    return rtypes.TypeSet(dim.has_missing, (ndim,), (in_type,), need_freeze=False)

@delayable()
def rset(source, dim=None):
    source = freeze(source)
    return apply_slice(source, "set", slices.AggregateSlice, dim, exec_set, check_set)

def binop(lsource, rsource, op, outtype=None):#{{{
    if(not isinstance(lsource, representor.Representor)):
        lsource = wrapper_py.rep(lsource)
    if(not isinstance(rsource, representor.Representor)):
        rsource = wrapper_py.rep(rsource)

    factive_left = lsource._active_slices
    factive_right = rsource._active_slices
    
    #check if number of slices match
    if(len(factive_left) != 1 and len(factive_right) != 1):
        assert (len(factive_left) == len(factive_right)), \
            "Number of slice do not match in binary operation " + op + \
            " on slices " + str(lsource._active_slices) + " and " + \
            str(rsource._active_slices)
    
    #equalize dim attributes
    assume_equal_dim = {}
    for fleft, fright in util.zip_broadcast(factive_left, factive_right):
        for d1, d2 in zip(fleft.dims, fright.dims):
            if(d1 != d2): 
                assume_equal_dim[(d2,)] = d1
    if(assume_equal_dim):
        rsource = redim(rsource, assume_equal_dim)

    #re-id slices with similar ids but non-similar slices 
    #(i.e. non-similar dimensions)
    rsource = realiasSimilarSlices(rsource, lsource._all_slices, 
                                            keep_equal=True)
    factive_right = rsource._active_slices
    nall_slices = lsource._all_slices.copy()
    nall_slices.update(rsource._all_slices)
    
    #create calc slices
    nactive_slices = []
    for pos, (fleft, fright) in \
                enumerate(util.zip_broadcast(factive_left, factive_right)):
        nslice = slices.BinElemOpSlice(fleft, fright, op, outtype, pos)
        nactive_slices.append(nslice)
        nall_slices[nslice.id] = nslice


    self = MultiOpRep((lsource, rsource), nall_slices, 
            tuple(nactive_slices))
    return self#}}}

def unary_op(source, op, outtype=None): #{{{
    nactive_slices = []                       #new active slices
    
    for slice in source._active_slices:
        nslice = slices.UnaryElemOpSlice(slice, op, outtype)
        nactive_slices.append(nslice)

    all_slices = source._all_slices.copy()
    for slice in nactive_slices:
        all_slices[slice.id] = slice

    #initialize object attributes
    self = UnaryOpRep((source,), all_slices, 
        tuple(nactive_slices))
    return self 
#}}}

class Group(MultiOpRep):
    def __init__(self, source, groupconstraint, keepers={}, name=None):
        groupslices = groupconstraint._active_slices
        if(len(groupslices) > 1):
            sdims = set([slice.dims for slice in groupslices])
            assert len(sdims) == 1, "Dimensions of multiple group fields should be the same"
            togroup_dims = sdims.pop()
        else:
            togroup_dims = groupslices[0].dims
       
        odim = togroup_dims[-1]
        group_dims = [dimensions.Dim(UNDEFINED, odim.variable, odim.has_missing, name="g" + slice.name) 
                 for slice in groupslices]
        
        if(not name is None):
            inner_dim = dimensions.Dim(UNDEFINED, len(group_dims), False, name=name)
        else:
            inner_dim = dimensions.Dim(UNDEFINED, len(group_dims), False)
    
        match_slices, start_depths = dim_helpers.matchDimPath(source._all_slices.values(), togroup_dims)
       
        keep_set = set()
        nkeepers = defaultdict(set)
        for keep, keep_slices in keepers.iteritems():
            keep_slices = set([slice.id for slice in source.get(*keep_slices)._active_slices])
            assert not keep_slices & keep_set, "Slice occuring multiple times in kept slices for grouping"
            keep_set.update(keep_slices)
            nkeepers[keep] = keep_slices
        
        for pos, slice in enumerate(groupslices):
            if(slice.id in keep_set):
                continue
            keep_set.add(slice.id)
            nkeepers[pos].add(slice.id)
        nkeepers["__group__"] = set([slice.id for slice in match_slices]) - keep_set
        
        
        nall_slices = source._all_slices.copy()
        keepslices = {}
        for keep, keep_slices in nkeepers.iteritems():
            if(keep == "__group__"):
                new_dimpath = togroup_dims[:-1] + tuple(group_dims) + (inner_dim,)
                varadapt = (len(group_dims),) #last dim is replaced by 1 + len(groupslices) dims
            else:
                if(isinstance(keep, tuple)):
                    kgroup_dims = tuple([group_dims[k] for k in keep])
                else:
                    kgroup_dims = (group_dims[keep],)
                    keep = (keep,) 

                new_dimpath = togroup_dims[:-1] + kgroup_dims
            
                varadapt = (len(kgroup_dims) - 1,) #last dim is replaced by 1 + len(groupslices) dims
            
            sel_slices = []
            sel_start_depths = []
            for sd, slice in zip(start_depths, match_slices):
                if(slice.id in keep_slices):
                    sel_slices.append(slice)
                    sel_start_depths.append(sd)

            nslices, startdepths = dim_helpers.redimMatch(sel_slices, sel_start_depths, togroup_dims, new_dimpath, varadapt)
            
            for slice in nslices:
                nall_slices[slice.id] = slice
            keepslices[keep] = (nslices, startdepths, new_dimpath)

        nactive_slices = [nall_slices[slice.id] for slice in source._active_slices] 

        self._group_slices = groupslices
        self._keep_slices = keepslices

        MultiOpRep.__init__(self, (source, groupconstraint), nall_slices, 
            tuple(nactive_slices))
      
class Stack(MultiOpRep):
    def __init__(self, sources, dims=None):
        if(isinstance(dims, tuple)):
            dimpaths = [dim_helpers.identifyDimPath(source, dim) for source, dim in zip(sources, dims)]
        else:
            dimpaths = [dim_helpers.identifyDimPath(source, dims) for source in sources]

        assert all([isinstance(dpath, tuple) and len(dpath) > 0 for dpath in dimpaths]), \
                                "Could not find valid dimpath in one or more of stack sources"

        bpaths = [dpath[1:] for dpath in dimpaths]
        bplan, final_dims = planBroadcast(*bpaths)
        

        startdepths = []
        match_slices = []
        for dpath, source in zip(dimpaths, sources):
            mslices, sdepths = dim_helpers.matchDimPath(source._all_slices.values(), dpath)
            match_slices.append(mslices)
            startdepths.append(sdepths)
 
        MultiOpRep.__init__(self, (source, groupconstraint), nall_slices, 
            tuple(nactive_slices))


@delayable()
class flat(UnaryOpRep):
    def __init__(self, source, dim_selector=-1):
        fdim = dim_helpers.identifyDimPath(source, dim_selector)
        
        assert len(fdim) > 1, "Cannot flatten a non-nested dim"
        
        
        multi_slices, startdepths = dim_helpers.matchDimPath(source._all_slices.values(), fdim[:-1])

        match_slices, startdepths = dim_helpers.matchDimPath(multi_slices, fdim)

        nslices, startdepths = dim_helpers.redimMatch(match_slices, startdepths, fdim, fdim[:-1], (-1,))

        nall_slices = source._all_slices.copy()
        for slice in nslices:
            nall_slices[slice.id] = slice

        nactive_slices = [nall_slices[slice.id] for slice in source._active_slices]

        UnaryOpRep.__init__(self, (source, ), nall_slices, 
            tuple(nactive_slices))
        
        
        self._flat_slices = set([slice.id for slice in nslices])
        self._flat_depths = startdepths
        self._flat_dim = fdim

from wrappers import wrapper_py
