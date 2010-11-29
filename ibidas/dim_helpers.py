import itertools
import operator

_delay_import_(globals(),"itypes","dimensions")
_delay_import_(globals(),"slices")

def maxmatchPath(pos_paths, match_path):#{{{
    """Find the maximum match of `match_path` in `pos_paths`

    Parameters:
    match_path: a tuple of dimensions
    pos_paths: a sequence of tuples of dimensions

    Returns:
    Returns tuple with (path matched, broadcast plan, number of matches)
    """
    if(not match_path or not pos_paths):
        return ((),(), 0)

    cur_matched = -1
    for path in pos_paths:
        plan, fdim = planBroadcast(path, match_path)
        matched = len(planPattern(plan,(0,1)))
        if(matched > cur_matched):
            cur_path = path
            cur_plan = plan
            cur_matched = matched
    
    pos = 0
    while(pos < len(cur_plan) and cur_plan[pos] == 1):
        pos += 1
    
    return (cur_path, cur_plan, cur_matched)#}}}

def planPos(plan, nr):#{{{
    """Generates the positions of the bit flags
    in a plan for source path(s) `nr`. To each 
    position, `startval` is added.
    
    
    Parameters
    ----------
    nr: (sequence of) nr of source path checked in plan

    Returns
    -------
    tuple with position of matches
    """
    if(operator.isSequenceType(nr)):
        match = 0
        for n in nr:
            match |= 2**n
    else:
        match = 2**nr
    assert match, "Match or nr parameter should be given."
    pattern = []
    for curpos, p in enumerate(plan):
        if(p & match == match):
            pattern.append(curpos)
    return tuple(pattern)#}}}

def planPattern(plan, nr, startval=0):#{{{
    """Generates the positions of the bit flags
    in a plan for source path(s) `nr`, EXCLUDING the
    matched plan actions. So it counts for each matched 
    action the number of non-matched actions previous to it.
    To each position,`startval` is added.
    
    
    Parameters
    ----------
    nr: (sequence of) nr of source path checked in plan

    Returns
    -------
    tuple with for each match the number of non-matches prepending it
    """
    if(operator.isSequenceType(nr)):
        match = 0
        for n in nr:
            match |= 2**n
    else:
        match = 2**nr
    assert match, "Match or nr parameter should be given."
    pattern = []
    curpos = startval
    for p in plan:
        if(p & match == match):
            pattern.append(curpos)
        else:
            curpos += 1
    return tuple(pattern)#}}}
        
def planBroadcast(*paths):#{{{
    """Matches a set of paths, generating a broadcast plan
    and a combined dimpath.
    
    Parameter
    ---------
    paths: dimension paths

    Returns
    -------
    a tuple of a plan and the combined dimpath.
    plan: a tuple of actions, equal in length to dimpath.
          each source dimpath has a bit-flag, indicating if it should
          be broadcasted or split in the action. 
    """
    final_dims = []
    plan = []
    npaths = [list(path) for path in paths]
    _planBroadcastHelper(npaths, final_dims, plan)
    return (tuple(plan), tuple(final_dims))#}}}

def _planBroadcastHelper(paths, final_dims, plan):#{{{
    curcol = len(paths) - 1
    while(curcol):
        if(not paths[curcol]):
            curcol -=1
            continue
        
        cdim = paths[curcol][0]
        cplan = 2 ** curcol

        match_paths = [[]] * len(paths)
        recursive=False
        for ccol in xrange(curcol):
            try:
                npos = paths[curcol].index(cdim)
                if(npos > 0):
                    match_paths.append(paths[curcol][:npos]) 
                    recursive=True
                cplan |= 2 ** ccol
                paths[ccol] = paths[ccol][npos:]
            except ValueError:
                pass
        if(recursive):
            _planBroadcastHelper(match_paths,final_dims,plan)
        
        if(cdim.variable):
            var = dim.variable
            pos = 0
            while(var):
                pos += 1
                if(plan[-pos] & 2**curcol):
                    var -=1
            if(cdim.variable != pos):
                cdim = cdim.copy(reid=True)
                cdim.variable = pos
        final_dims.append(cdim)
        plan.append(cplan)
#}}}

def pathSuffix(dimpath, minlen = 1): #{{{
    """Returns those dimensions of dimpath that can stand alone
    from earlier dimensions in the path (i.e. are not dependent on).

    Parameters
    ----------
    dimpath: tuple of dimensions
    minlen: minimum length of the suffix
    
    Returns
    -------
    tuple of prefix path and suffix path
    """
    need = minlen
    pos = 0
    for pos in xrange(len(dimpath)):
        need = max(dimpath[-(pos + 1)].variable, need - 1)
        if(need == 0):
            break
    return (dimpath[:-(pos + 1)], dimpath[-(pos + 1):])#}}}

def broadcastAndMatchDimPath(slices, path_prefix, path_suffix, single_match_suffix=False):#{{{
    """Matches two sections of a dimpath, of which the first can be 
    broadcasted and the second has to be matched exactly.

    Parameters
    ----------
    slices: slices on which to perform the matching
    path_prefix: part of dim path that can be broadcasted
    path_suffix: part of dim path that should match exactly
    single_match_suffix: only the last dim of the suffix should not take part in the broadcasting,
                         but matched slices should contain whole suffix. 
    
    Returns
    -------
    Filter path with best (max) match
    """
    #find slices that match dimpath suffix, find prev_paths for these matches
    pos_slices, dummy, prev_paths = matchDimPath(slices, path_suffix, return_prev=True)
    assert pos_slices, "Cannot find any slice that matches dim path suffix"
    
    #last dimension is matched by array or slice
    #find max match with all possible prev paths of this last dimension
    
    #first adapt prev_paths if prefix of the suffix can also be broadcasted on
    prev_paths = set(prev_paths)
    if(single_match_suffix):
        prev_paths = [ppath + path_suffix[:-1] for ppath in prev_paths]
        path_suffix = path_suffix[-1:]
    
    #find max match path
    prev_path, prev_plan, prev_matched = maxmatchPath(prev_paths, path_prefix)

    #select suffix containing matched part if necessary
    if(prev_matched > 0):
        first_match_pos = planPos(prev_plan, 1)[0]
        if(first_match_pos > 0):
            prev_path = pathSuffix(prev_path, len(prev_path) - first_match_pos)[1]

    #construct filter path 
    filter_dimpath = prev_path + path_suffix
    return filter_dimpath #}}}

def matchDimPath(slices, dimpath, return_prev=False):#{{{
    """Matches dimpath exactly to dim paths in slices.
       Returns tuple containing a list of matched slices and 
       a list containing tuples of matched start positions for each slice
       if return_prev is given, returns as third parameter the header dim paths in
       a set.
    """
    match_slices = []
    start_depths = []
    ret_prev = []
    ldpath = len(dimpath)
    for slice in slices:
        pos = 0
        sdims = slice.dims
        lsdims = len(sdims)
        startpos = []
        while(pos < lsdims and (lsdims - pos) >= ldpath):
            try:
                curstart = sdims.index(dimpath[0], pos)
            except ValueError:
                break
            if(sdims[(curstart + 1):(curstart + ldpath)] != dimpath[1:]):
                pos = curstart + 1
            else:
                startpos.append(curstart)
                pos = curstart + len(dimpath)
                if(return_prev):
                    ret_prev.append(sdims[:curstart])
        if(startpos):
            match_slices.append(slice)
            start_depths.append(tuple(startpos))

    if(return_prev):
        return (tuple(match_slices), tuple(start_depths), ret_prev)
    else:
        return (tuple(match_slices), tuple(start_depths))#}}}

def redimMatch(match_slices, start_depths, oldpath, newpath, var_adapt=(0,)):#{{{
    """
    Takes output of matchDimPath, and adapts oldpath to newpath.
    Returns new slices and new start posses.
    """
    nslices = []
    loldpath = len(oldpath)
    dep_redim_cache = {}
    lendiff = len(newpath) - loldpath
    nstart_depths = []

    for slice, startposs in zip(match_slices, start_depths):
        slice = slice.copy()
        clendiff = 0  #current dim pos diff. (caused by exchanging oldpath to newpath)
        lstartposs = len(startposs)
        if(lstartposs > 1):
            nstartposs = []
        for startpos in startposs:
            startpos += clendiff
            clendiff += lendiff
            if(lstartposs > 1):
                nstartposs.append(startpos)
            
            #dependent part of the dimensions
            dep = slice.dims[(startpos + loldpath):]

            #modify dependent part
            ndep = []
            for pos, dim in enumerate(dep):
                if dim.variable - pos > 0: #falls dependency within exchanged part?
                    if(dim in dep_redim_cache): #look for new dim in redim cache
                        ndep.append(dep_redim_cache[dim])
                    else:                       #otherwise, create it. var_adapt param tells us how to handle it.
                        ndim = dim.copy(reid=True)
                        ndim.variable = max(ndim.variable + var_adapt[min(dim.variable - pos - 1, len(var_adapt) - 1)],0)
                        ndep.append(ndim)
                        dep_redim_cache[dim] = ndim
                else:
                    ndep.append(dim)
            dep = tuple(ndep)

            slice.dims = slice.dims[:startpos] + newpath + dep
            slice.type = slice.type.redim_var(len(dep), var_adapt, dep_redim_cache)
            
            ldep = len(dep)
        if(lstartposs > 1):
           nstart_depths.append(tuple(nstartposs))
        else:
           nstart_depths.append(startposs)
        nslices.append(slice)
    return (tuple(nslices), tuple(nstart_depths))#}}}

def commonDimPath(source):#{{{
    """Returns common dimensions shared by all slices"""
    if(len(source._active_slices) == 0):
        return ()

    pos = 0
    minlen = min([len(slice.dims) for slice in source._active_slices])
    while(pos < minlen):
        cdim = set([slice.dims[pos] for slice in source._active_slices])
        pos += 1
        if(len(cdim) != 1):
            break

    return source._active_slices[0].dims[:pos]#}}}

def uniqueDimPath(source):#{{{
    """Returns unique dim path, i.e. at each nesting level determines
    if dim is unique and adds it to path. If dim is not unique, returns False"""

    fdims = set([slice.dims for slice in source._active_slices 
                            if slice.dims])
    if(len(fdims) <= 1):
        path = fdims.pop()
    else:
        path = []
        for elems in itertools.izip_longest(*fdims):
            dimset = set([elem for elem in elems if not elem is None])
            if(len(dimset) == 1):
                path.append(dimset.pop())
            else:
                return False
        path = tuple(path)
     
    return path#}}}

def sliceHasDimPath(slice, dim_path):#{{{
    sdims = slice.dims
    pos = -1
    try:
        for dim in dim_path:
            pos = sdims.index(dim, pos + 1)
    except ValueError:
        return False
    return npos#}}}

def identifyDimPath(source, dim_selector=None):#{{{
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
            search for dim with name, use it as selector
            search for slice with name: if found, use slice as selector
        
        if dim: find path, ending with dim, that has all its dependencies
        if slice: use its dims as selector
        if tuple of dim (names):
            
        if tuple of dim (names):
            if empty: return False
            else, matches dims in tuple:
            Starts from last dim, find unique path to fixed dim.
        if int:
            if there is a unique dim path, use that to find index dim.
    """
    if(isinstance(dim_selector, tuple) and len(dim_selector) == 0):
        dim_selector = None

    if(dim_selector is None):
        res = commonDimPath(source)
        return identifyDimPathHelper(source, res)

    elif(isinstance(dim_selector, int)):
        path = uniqueDimPath(source)
        if(path is False):
            return False

        if(dim_selector < 0):
            dim_selector = len(path) + dim_selector
            assert dim_selector >= 0, "Unique dim path not long enough"

        return identifyDimPathHelper(source, (path[dim_selector],))

    elif(isinstance(dim_selector, str)):
        if(dim_selector[0].isupper()):
            if(dim_selector[0] == "D"):
                nselectors = source._active_dim_dict[dim_selector[1:]]
            elif(dim_selector[0] == "A"):
                nselectors = source._active_slice_dict[dim_selector[1:]]
            else:
                raise RuntimeError, "Unsupported Axis selector in dim_selector " + dim_selector
        else:
            if(dim_selector in source._active_dim_dict):
                nselectors = source._active_dim_dict[dim_selector]
            elif(dim_selector in source._active_slice_dict):
                nselectors = source._active_slice_dict[dim_selector]
            else:
                return False

        results = []
        for selector in nselectors:
            res = identifyDimPath(source, selector)
            if(res is True):
                return True
            elif(not res is False):
                results.append(res)


        if(len(results) == 1):
            return results[0]
        elif(len(results) == 0):
            return False
        else:
            results = set(results)
            if(len(results) == 1):
                return results.pop()
            else:
                return True

    elif(isinstance(dim_selector, dimensions.Dim)):
        return identifyDimPathHelper(source, (dim_selector,))
    elif(isinstance(dim_selector, slices.Slice)):
        return identifyDimPathHelper(source, dim_selector.dims)
    elif(isinstance(dim_selector, tuple)):
        return identifyDimPathHelper(source, dim_selector)
    else:
        raise RuntimeError, "Unexpected dim selector: " + str(dim_selector)#}}}

def identifyDimPathHelper(source, path):#{{{
    """Given a path (tuple of dims or dim names), 
    Determines if there is a matching dim path in source that uniquely 
    identifies the dimension last in path. Such a path should have the 
    same order of dimenmsions as given in 'path', but not necessarily the
    same dimensions. It however should be unique.
    
    Returns an dim path (tuple of dims) if unique path found.
    (See _identifyDim for description).

    Returns
    -------
    True: if multiple paths are possible
    False: if no path is matching
    dimension identifier otherwise
    """
    if(not path):
        return False

    reslist = []

    done = False
    for pos, dim in enumerate(path[::-1]):
        #make certain dim is an (available dim)
        if(isinstance(dim, dimensions.Dim)):
            if not dim.id in source._active_dim_id_dict:
                return False
        else: #convert str to dim
            dims = source._active_dim_dict[dim]
            if(len(dims) > 1): #multiple matches for dim name. Try them all.
                sresults = []
                for dim in dims:
                    sres = identifyDimPathHelper(source, path[:(-(pos + 1))] + (dim,))
                    if(sres is True):
                        return True
                    elif(not sres is False):
                        sresults.append(sres)
                if(len(sresults) > 1):
                    return True
                elif(len(sresults) == 0):
                    return False
                reslist.extend(sresults[0][::-1]) #only one result, add dim path to reslist
                break #all matching was done recursively, we are done here
            elif(len(dims) == 1):
                dim = iter(dims).next()
            else:
                return False
                
        if(not reslist): 
            reslist.append(dim)
        else:
            while(True):
                parentdims = source._active_dim_id_parent_dict[reslist[-1].id]
                if(dim in parentdims):
                    reslist.append(dim)
                    break
                #if dim not in parentdims, we have to add intermediate dims... 
                if(len(parentdims) > 1): #multiple paths possible, do it recursively
                    sresults = []
                    for pdim in parentdims:
                        sres = identifyDimPathHelper(source, 
                                        path[:(-(pos + 1))] + (pdim, dim))
                        if(sres is True):
                            return True
                        elif(not sres is False):
                            sresults.append(sres)
                    if(len(sresults) > 1):
                        return True
                    elif(len(sresults) == 0):
                        return False
                    reslist.extend(sresults[0][::-1])
                    done = True
                    break
                elif(len(parentdims) == 1):
                    reslist.append(iter(parentdims).next())
                else:
                    return False
            if(done is True):
                break
    
    #if dims in reslist are variable, we have to make certain their base dimensinos
    #are available
    add_remaining = 0
    for dim in reslist:
        add_remaining -= 1 
        add_remaining = max(add_remaining, dim.variable)

    #add remaining dimensions 
    while(add_remaining):
        add_remaining -= 1
        parentdims = source._active_dim_id_parent_dict[reslist[-1].id]
        if(len(parentdims) > 1):
            return True
        else:
            dim = iter(parentdims).next()
            if(dim is None):
                raise RuntimeError, "Cannot add needed remaining dimensions. Bug?"
            add_remaining = max(add_remaining, dim.variable)
            reslist.append(dim)
    return tuple(reslist[::-1])
#}}}
