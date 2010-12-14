import itertools
from collections import defaultdict
import numpy

_delay_import_(globals(),"dimensions","Dim")
_delay_import_(globals(),"..slices")

class DimPath(tuple):
    def __new__(cls, *dims):
        assert all([isinstance(dim, Dim) for dim in dims]),"DimPath should consist of dimensions"
        return tuple.__new__(cls, dims)
    
    def _getNames(self):
        return [dim.name for dim in self]
    names = property(fget=_getNames)

    def hasName(self,name):
        return any([dim.name == name for dim in self])

    def getDimByName(self,name):
        return self[self.names.index(name)]
    def getDimIndexByName(self,name):
        dimnames = self.names
        try:
            return dimnames.index(name)
        except ValueError:
            return None

    def __getslice__(self,*idx):
        res = tuple.__getslice__(self,*idx)
        if(isinstance(res,tuple)):
            return DimPath(*res)
        return res

    def __getitem__(self,*idx):
        res = tuple.__getitem__(self,*idx)
        if(isinstance(res,tuple)):
            return DimPath(*res)
        return res

    def __repr__(self):
        return "DimPath" + tuple.__repr__(self)
    

    def __add__(self, other):
        return DimPath(*tuple.__add__(self,other))

    def __contains__(self,part):
        if(isinstance(part,DimPath)):
            pos = -1
            try:
                for dim in dim_path:
                    pos = self.index(dim, pos + 1)
            except ValueError:
                return False
            return True
        else:
            return tuple.__contains__(self,part)

    def changeNames(self,newname_dict):
        newpath = []
        for dim in self:
            if(dim in newname_dict):
                dim = dim.copy()
                dim.name = newname_dict[dim]
            elif(dim.name in newname_dict):
                dim = dim.copy()
                dim.name = newname_dict[dim.name]
            newpath.append(dim)
        return DimPath(*newpath)

    def updateDimVariable(self,insertsize=1,insertpoint=0):
        ndims = []
        found = False
        for pos in xrange(len(self)):
            dim = self[pos]
            if(dim.variable > (pos - insertpoint)):
                found = True
                ndim = dim.copy(reid=False)
                ndim.variable += insertsize
            else:
                ndim = dim
            ndims.append(ndim)     
        if(not found): 
            return self
        else:
            return DimPath(*ndims)

    def contigiousFixedNDims(self):
        depth = 0
        #maximize array size with fixed dims
        while(depth < len(self) and not self[depth].variable and not 
                self[depth].has_missing):
            depth+=1
        return depth
    
    def matchDimPath(self, dimpath):#{{{
        """Matches dimpath exactly to dim paths in slices.
        Returns a list containing start positions
        """
        start_depths = []
        
        pos = 0
        ndims = len(self)
        nmpath = len(dimpath)
        startpos = []
        while(pos < ndims and (ndims - pos) >= nmpath):
            try:
                curstart = self.index(dimpath[0], pos)
            except ValueError:
                break
            if(self[(curstart + 1):(curstart + nmpath)] != dimpath[1:]):
                pos = curstart + 1
            else:
                startpos.append(curstart)
                pos = curstart + nmpath
        
        return startpos#}}}
   
def commonDimPath(dimpaths):#{{{
    """Returns common dimensions shared by all slices"""
    pos = 0
    minlen = min([len(dp) for dp in dimpaths])
    while(pos < minlen):
        cdim = set([dp[pos] for slice in dimpaths])
        pos += 1
        if(len(cdim) != 1):
            break
    return dimpaths[0][:pos]#}}}

def uniqueDimPath(dimpaths,only_complete=True):#{{{
    """Returns unique dim path, i.e. at each nesting level determines
    if dim is unique and adds it to path. """

    fdims = set(dimpaths)

    if(len(fdims) <= 1):
        path = fdims.pop()
    else:
        path = []
        for elems in itertools.izip_longest(*fdims):
            dimset = set([elem for elem in elems if not elem is None])
            if(len(dimset) == 1):
                path.append(dimset.pop())
            else:
                if(only_complete):
                    return False
                else:
                    break
        path = DimPath(*path)
     
    return path#}}}

def planBroadcastEqual(paths,partial=False):
    maxlen = max([len(path) for path in paths])
    plans = [[] * maxlen for path in paths]
    bcdims = [None] * maxlen

    curpos = -1
    while(curpos >= -maxlen):
        xdims = set()
        l = set()
        for path in paths:
            if(len(path) >= -curpos):
                xdims.add(path[curpos])
                l.add(path[curpos].shape)
            else:
                xdims.add(None)
        xdims = [path[curpos] for path in paths]


        if(len(l) > 1):
            l.discard(1)
        if(len(l) > 1):
            l.discard(UNDEFINED)
        assert len(l) == 1, "Different shaped dimensions cannot be broadcast to each other"
        length = l.pop()
        bcdim = [xdim for xdim in xdims if xdim.shape == length][0]
        for xdim,plan in zip(xdims,plans):
            if(xdim is None):
                plan[curpos] = BCNEW
            elif(xdim.shape == 1):
                plan[curpos] = BCEXIST
            elif(xdim != bcdim):
                plan[curpos] = BCENSURE
            else:
                plan[curpos] = BCCOPY
        bcdims[curpos] = bcdim 
        curpos -=1 
    return (bcdims, plans)        

def flatFirstDims(array,ndim):
    oshape = array.shape
    ndim = ndim + 1
    rshape = (int(numpy.multiply.reduce(array.shape[:ndim])),) + array.shape[ndim:]
    return array.reshape(rshape)

def createDimSet(sourcepaths):
    dimset = set()
    for dimpath in sourcepaths:
        dimset.update(dimpath)
    return dimset

def createDimParentDict(sourcepaths):
    parents = defaultdict(list)
    for dimpath in sourcepaths:
        parents[dimpath[0]].append(None)
        for pos in xrange(1,len(dimpath)):
            parents[dimpath[pos]].append(dimpath[pos-1])
    return parents


def identifyDimPath(sourcepaths, dim_selector=None):#{{{
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
        res = commonDimPath(sourcepaths)
        return identifyDimPathHelper(sourcepaths, res)

    elif(isinstance(dim_selector, int)):
        path = uniqueDimPath(sourcepaths,only_complete=(dim_selector < 0))
        if(path is False):
            return False

        return identifyDimPathHelper(sourcepaths, (path[dim_selector],))

    elif(isinstance(dim_selector, str)):
        nselectors = [dpath.getDimByName(name) for dpath in sourcepaths if dpath.hasName(name)]
        results = []
        for selector in set(nselectors):
            res = identifyDimPath(sourcepaths, selector)
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

    elif(isinstance(dim_selector, Dim)):
        return identifyDimPathHelper(sourcepaths, (dim_selector,))
    elif(isinstance(dim_selector, slices.Slice)):
        return identifyDimPathHelper(sourcepaths, dim_selector.dims)
    elif(isinstance(dim_selector, tuple)):
        return identifyDimPathHelper(sourcepaths, dim_selector)
    else:
        raise RuntimeError, "Unexpected dim selector: " + str(dim_selector)#}}}

def identifyDimPathHelper(sourcepaths, path):#{{{
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

    #create set and dict of dims
    dimset = createDimSet(sourcepaths)
    dimparents = createDimParentDict(sourcepaths)

    done = False
    for pos, dim in enumerate(path[::-1]):
        #make certain dim is an (available dim)
        if(isinstance(dim, Dim)):
            if not dim in dimset:
                return False
        else: #convert str to dim
            dims = [d for d in dimset if d.name == dim]
            if(len(dims) > 1): #multiple matches for dim name. Try them all.
                sresults = []
                for dim in dims:
                    sres = identifyDimPathHelper(sourcepaths, path[:(-(pos + 1))] + (dim,))
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
                pdims = dimparents[reslist[-1]]
                if(dim in pdims):
                    reslist.append(dim)
                    break
                #if dim not in pdims, we have to add intermediate dims... 
                if(len(pdims) > 1): #multiple paths possible, do it recursively
                    sresults = []
                    for pdim in pdims:
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
                elif(len(pdims) == 1):
                    reslist.append(iter(pdims).next())
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
        pdims = dimparents[reslist[-1]]
        if(len(pdims) > 1):
            return True
        else:
            dim = iter(pdims).next()
            if(dim is None):
                raise RuntimeError, "Cannot add needed remaining dimensions. Bug?"
            add_remaining = max(add_remaining, dim.variable)
            reslist.append(dim)
    return tuple(reslist[::-1])
#}}}
