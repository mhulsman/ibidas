import itertools
from collections import defaultdict
import numpy

from ..constants import *
from dimensions import Dim
#_delay_import_(globals(),"dimensions","Dim")
_delay_import_(globals(),"rtypes")
_delay_import_(globals(),"..representor")
_delay_import_(globals(),"..utils","toposort","util","context")

class PathError(Exception):
    pass

class DimPath(tuple):
    def __new__(cls, *dims):
        assert all([isinstance(dim, Dim) for dim in dims]),"DimPath should consist of dimensions"
        return tuple.__new__(cls, dims)
   
    def __reduce__(self):
        return (DimPath, tuple(self))

    def _getNames(self):
        return [dim.name for dim in self]
    names = property(fget=_getNames)
    
    def _getIDs(self):
        return [dim.id for dim in self]
    ids = property(fget=_getIDs)

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

    def getDimIndices(self, selector):
        if isinstance(selector,int):
            return (selector,)
        elif isinstance(selector,str):
            return (self.getDimIndexByName(selector),)
        elif isinstance(selector,tuple):
            return sum([self.getDimIndices(elem) for elem in selector],())

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

    def strip(self):
        if(self and self[0] is root):
            if(self[-1] is end):
                return self[1:-1]
            else:
                return self[1:]
        else:
            if(self and self[-1] is end):
                return self[:-1]
            else:
                return self

    def __contains__(self,part):#{{{
        if(isinstance(part,DimPath)):
            if(not part):
                return False
            elif(part[0] is root):
                if(part[-1] is end):
                    return self == part[1:-1]
                else:
                    return self[:(len(part)-1)] == part[1:]
            elif(part[-1] is end):
                return self[-(len(part)-1):] == part[:-1]
            else:
                pos = 0
                ndims = len(self)
                nmpath = len(part)
                while(pos < ndims and (ndims - pos) >= nmpath):
                    try:
                        curstart = self.index(part[0], pos)
                    except ValueError:
                        return False

                    if(self[(curstart + 1):(curstart + nmpath)] == part[1:]):
                        return True
                    else:
                        pos = curstart + 1
                return False
        else:
            return tuple.__contains__(self,part)#}}}

    def changeNames(self,newname_dict):#{{{
        newpath = []
        for dim in self:
            if(dim in newname_dict):
                dim = dim.copy()
                dim.name = newname_dict[dim]
            elif(dim.name in newname_dict):
                dim = dim.copy()
                dim.name = newname_dict[dim.name]
            newpath.append(dim)
        return DimPath(*newpath)#}}}
    
    def removeDim(self, pos, elem_specifier=None, subtype=None):#{{{
        ndims = []
        for p in xrange(max(pos + 1,0), len(self)):
            r = self[p].removeDepDim(p - pos - 1, elem_specifier)
            ndims.append(r)

        res = self[:max(pos,0)] + DimPath(*ndims)
        if(not subtype is None):
            subtype = subtype._removeDepDim(pos=pos - len(self), elem_specifier=elem_specifier)
            return (res,subtype)
        else:
            return res#}}}

    def updateDim(self, pos, ndim, subtype=None):#{{{
        if(not isinstance(ndim,tuple)):
            ndim = (ndim,)
        
        ndims = []
        for p in xrange(max(pos + 1,0), len(self)):
            ndims.append(self[p].updateDepDim(p - pos - 1, ndim))
      
        if pos >=0:
            res = self[:max(pos,0)] + ndim + DimPath(*ndims)
        else:
            res = DimPath(*ndims)

        if(not subtype is None):     
            subtype = subtype._updateDepDim(pos=pos - len(self), ndim=ndim)
            return (res,subtype)
        else:
            return res#}}}
    
    def insertDim(self, pos, ndim, subtype=None):#{{{
        ndims = []
        if(pos < 0):
            refpos = pos;
        else:
            refpos = pos - 1;

        for p in xrange(max(pos,0), len(self)):
            ndims.append(self[p].insertDepDim(p - refpos - 1, ndim))
        if pos >= 0:
            res = self[:pos] + (ndim,) + DimPath(*ndims)
        else:
            res = DimPath(*ndims)
        if(not subtype is None):
            subtype = subtype._insertDepDim(pos=refpos - len(self), ndim=ndim)
            return (res,subtype)
        else:
            return res#}}}

    def permuteDims(self,permute_idxs, subtype=None, prevdims=tuple()):#{{{
        ndims = []
        if(prevdims):
            permute_start = len(permute_idxs) - len(prevdims)
            for pos, dim in enumerate(self):
                dep = dim.dependent
                if(len(dep) > pos - permute_start):
                    deppos = util.select(range(-len(prevdims),pos)[::-1],dep)
                    ndep = [False] * (len(prevdims) + pos)
                    redim=False
                    for dp in deppos:
                        if dp < permute_start:
                            ndep[permute_idx.index(dp + len(prevdims))] = True
                            redim=True
                        else:
                            ndep[dp + len(prevdims)] = True
                    if(redim):
                        dim = dim.changeDependent(ndep[::-1],(prevdims + tuple(ndims))[::-1])
                ndims.append(dim)
        else:
            for pos, permute_idx in enumerate(permute_idxs):
                dim = self[permute_idx]
                dep = dim.dependent
                if(dep):
                    deppos = util.select(range(0,permute_idx)[::-1],dep)
                    ndep = [False] * pos
                    for dp in deppos:
                        assert dp in permute_idxs[:pos], "Dependent dim " + str(self[dp]) + " of dim " + str(dim) + " cannot be placed after it"
                        ndep[permute_idxs.index(dp)]=True
                    dim = dim.changeDependent(ndep[::-1],ndims[::-1])
                ndims.append(dim)
        res = DimPath(*ndims)
        if(not subtype is None):
            subtype = subtype._permuteDepDim(prevdims=res, permute_idxs=permute_idxs)
            return (res,subtype)
        else:
            return res#}}}

    def contigiousFixedNDims(self):#{{{
        """Returns number of contigious non-variable non-missing dims from 
        start of path"""
        depth = 0
        #maximize array size with fixed dims
        while(depth < len(self) and not self[depth].isVariable() and not 
                self[depth].has_missing):
            depth+=1
        return depth#}}}
    
    def matchDimPath(self, dimpath):#{{{#{{{
        """Matches dimpath exactly to dim paths in self.
        Returns a list containing positions of last dim
        """
        lastpos = []
        nseldims = len(dimpath.strip())
        if(dimpath and dimpath[0] is root):
            if(dimpath[-1] is end):
                if(self == dimpath[1:-1]):
                    lastpos.append(nseldims - 1)
            else:
                if(self[:(len(dimpath)-1)] == dimpath[1:]):
                    lastpos.append(nseldims - 1)
        elif(dimpath and dimpath[-1] is end):
            if(self[-(len(dimpath)-1):] == dimpath[:-1]):
                lastpos.append(len(self) - 1)
        elif(dimpath):
            pos = 0
            ndims = len(self)
            while(pos < ndims and (ndims - pos) >= nseldims):
                try:
                    curstart = self.index(dimpath[0], pos)
                except ValueError:
                    break
                if(self[(curstart + 1):(curstart + nseldims)] != dimpath[1:]):
                    pos = curstart + 1
                else:
                    lastpos.append(curstart + nseldims - 1)
                    pos = curstart + nseldims
        
        return lastpos#}}}#}}}

    def completePath(self):#{{{
        for pos in xrange(len(self)):
            if(len(self[pos]) > pos):
                return False
        return True#}}} 
                
    def splitSuffix(self,minlength=1):#{{{
        assert minlength <= len(self), "Minlength too large in splitSuffix"

        rev_self = self[::-1]
        rev_suffix = []
        curpos = 0
        while len(rev_suffix) < minlength:
            assert curpos < len(rev_self), "Cannot find complete suffix"
            dim = rev_self[curpos]
            minlength = max(len(dim) + curpos,minlength)
            rev_suffix.append(dim)
                
        return DimPath(*rev_suffix[::-1])                #}}}

def commonDimPath(dimpaths):#{{{
    """Returns common dimensions shared by all slices"""
    pos = 0
    minlen = min([len(dp) for dp in dimpaths])
    while(pos < minlen):
        cdim = set([dp[pos] for dp in dimpaths])
        if(len(cdim) != 1):
            break
        pos += 1
    return list(dimpaths)[0][:pos]#}}}

def uniqueDimPath(dimpaths,only_unique=True):#{{{
    """Returns unique dim path, i.e. at each nesting level determines
    if dim is unique and adds it to path. """

    if not dimpaths:
        return []
    fdims = set(dimpaths)

    if(len(fdims) == 1):
        path = fdims.pop()
    else:
        path = []
        for elems in itertools.izip_longest(*fdims):
            dimset = set([elem for elem in elems if not elem is None])
            if(len(dimset) == 1):
                path.append(dimset.pop())
            elif only_unique:
                return path
            else:
                path.append(dimset)
    return path#}}}

def planBroadcastMatchPos(paths):#{{{
    """Matches dims in paths based on their position.
    Returns new set of dims and broadcast plan."""

    maxlen = max([len(path) for path in paths])
    plans = [[] for path in paths]
    bcdims = [None] * maxlen
    curpos = - 1
    while(-(curpos + 1) < maxlen):
        xdims = []
        l = set()
        for path in paths:
            if(len(path) > -(curpos+1)):
                xdims.append(path[curpos])
                l.add(path[curpos].shape)
            else:
                xdims.append(None)

        if(len(l) > 1):
            l.discard(1)
        if(len(l) > 1):
            l.discard(UNDEFINED)
        assert len(l) == 1, "Different shaped dimensions cannot be broadcast to each other"
        length = l.pop()
        nxdims = len(set(xdims))
        bcdim = [xdim for xdim in xdims if not xdim is None and xdim.shape == length][0]
        for xdim,plan in zip(xdims,plans):
            if(xdim is None):
                plan.append(BCNEW)
            elif(xdim.shape == 1 and bcdim.shape != 1):
                plan.append(BCEXIST)
            elif(xdim != bcdim):
                plan.append(BCENSURE)
            elif nxdims == 1:
                plan.append(BCCOPY)
            else:
                plan.append(BCSOURCE)
        bcdims[curpos] = bcdim 
        curpos -=1 
    for pos in xrange(len(plans)):
        plans[pos] = plans[pos][::-1]
    
    return (bcdims, plans)        #}}}

def planBroadcastMatchDim(paths):#{{{
    """Matches dims in paths based on their identity, as well
    as their potential to be used as broadcast dimension (shape == 1).
    Returns new set of dims and broadcast plan"""
    graph = toposort.StableTopoSortGraph()

    #a dim can occur multiple times in dag, to prevent cycles
    #Therefore, it has to be represented by ids instead of itself
    #Step 1: assign ids to dims
    curid = 0
    translate = {}

    dimnames = dict()
    for dim in itertools.chain(*paths):
        if not dim in translate:
            translate[dim] = [curid]
            curid += 1
            if(dim.name in dimnames and dimnames[dim.name] != dim):
                raise RuntimeError, "Cannot broadcast, there are different dimensions with same name: " + str(dim.name) + ". Please rename (dim_rename) or state their equivalence."
            dimnames[dim.name] = dim
    
    #broadcast dim mergers
    wildcard_links = dict()
    free_wildcards = set()
    
    #Step 2: create graph
    for path in paths:
        lastid = None
        for pathpos, dim in enumerate(path[::-1]):
            dimid = translate[dim][0]
            newnode = False

            #new wildcard dim?
            if(False and dim.shape == 1 and dim.has_missing is False and not dimid in wildcard_links):
                odimid = dimid
                dimids = graph.getParents(lastid)
                if(dimids):
                    dimid = min(dimids) #maps to first ordered dim (i.e firstmost path)
                    wildcard_links[odimid] = dimid
                else:
                    graph.addNodeIfNotExist(dimid)
                    free_wildcards.add(dimid)
            elif(dimid in wildcard_links): #old wildcarddim, use previous mapping
                dimid = wildcard_links[dimid]
            else: #normal dim, ensure it available in the graph
                newnode = graph.addNodeIfNotExist(dimid)

            if(pathpos > 0):
                pos = len(translate[dim])
                #try to add edge, such that we get no cycle
                #if cycle is found, add new ids for target node
                while True:
                    try:
                        graph.addEdge(dimid,lastid)
                        break
                    except toposort.CycleError, e:
                        pos -= 1
                        if(pos < 0):
                            translate[dim].append(curid)
                            newnode = graph.addNodeIfNotExist(curid)
                            curid += 1
                            pos = -1
                        dimid = translate[dim][pos]
            
            if(free_wildcards and newnode):
                for nodeid in list(graph)[::-1]:
                    if(nodeid in free_wildcards and not nodeid == dimid):
                        try:
                            graph.mergeNodes(nodeid,dimid)
                            wildcard_links[nodeid] = dimid
                            free_wildcards.discard(nodeid)
                            break
                        except toposort.CycleError, e:
                            pass #cannot be merged, let it be
                    
            lastid = dimid 
  
    #Step 3: get ordered bc dims
    rev_translate = dict(sum([[(subid,dim) for subid in id] for dim,id in translate.iteritems()],[]))
    bcdims = list(graph)
    bcdims = [rev_translate[bcdimid] for bcdimid in bcdims]

    #Step 4: create plans for each path
    plans = []
    for path in paths:
        plan = []
        pathpos = len(path) - 1
        for bcdim in bcdims[::-1]:
            if(pathpos >= 0):
                if( bcdim == path[pathpos]):
                    plan.append(BCCOPY)
                    pathpos -= 1
                else: 
                    for dimid in translate[path[pathpos]][::-1]:
                        if(dimid in wildcard_links):
                            bdim = rev_translate[wildcard_links[dimid]]
                            if(bdim == bcdim):
                                plan.append(BCEXIST)
                                pathpos -= 1
                                break
                    else:
                        plan.append(BCNEW)
            else:
                plan.append(BCNEW)

        plans.append(plan[::-1])
    
    #Step 5:discriminate between BCCOPY and BCSOURCE
    for planpos in xrange(len(plans[0])):
        x = set([plan[planpos] for plan in plans])
        if BCCOPY in x and len(x) > 1:
            for plan in plans:
                if plan[planpos] == BCCOPY:
                    plan[planpos] = BCSOURCE

    return (bcdims,plans)#}}}

def planBroadcastFromPlan(path, plan, origdims, bcdims):
    plan = plan[::-1]
    bcdims = bcdims[::-1]
    ipath = path[::-1]
    origdims = origdims[::-1]

    existpos = [None] * len(plan)
    pathpos = 0
    origdimpos = 0
    
    #determine original dim positions
    for pos, (planelem, bcdim) in enumerate(zip(plan, bcdims)):
        if planelem == BCEXIST or planelem == BCENSURE:
            try: 
                pathpos = ipath.index(origdims[origdimpos], pathpos)
                existpos[pos] = pathpos
            except ValueError:
                pass
            origdimpos += 1
        elif planelem == BCCOPY or planelem == BCSOURCE:
            try:
                pathpos = ipath.index(bcdim, pathpos)
                existpos[pos] = pathpos
            except ValueError:
                pass
            origdimpos += 1
     
    #create new plan
    nplan = []
    nbcdims = []
    pathpos = 0
    for pos, (planelem, bcdim) in enumerate(zip(plan, bcdims)):
        nposses = [x for x in existpos[pos:] if not x is None]
        if nposses:
            nextpos = nposses[0]
        else:
            nextpos = len(ipath)
        if planelem == BCNEW:
            if pathpos < nextpos and bcdim in ipath[pathpos:nextpos]:
               xpos = ipath.index(bcdim, pathpos, nextpos)
               while pathpos <= xpos:
                    nplan.append(BCSOURCE)
                    nbcdims.append(ipath[pathpos])
                    pathpos += 1
            else:
                nplan.append(BCNEW)
                nbcdims.append(bcdim)
        elif planelem == BCEXIST or planelem == BCENSURE or planelem == BCCOPY or planelem == BCSOURCE:
            if existpos[pos] is None:
                nplan.append(BCNEW)
                nbcdims.append(bcdim)
            else:
                while pathpos < existpos[pos]:
                    nplan.append(BCSOURCE)
                    nbcdims.append(ipath[pathpos])
                    pathpos += 1
                nplan.append(planelem)
                nbcdims.append(bcdim)
                pathpos += 1
        else:
            raise RuntimeError, "Unknown path element"
       
    return (nbcdims[::-1],nplan[::-1])

def processPartial(bcdims, plans):
    nplans = list(plans)
    for i in range(len(bcdims)):
        pos = -(i+1)
        if bcdims[pos].dependent:
            break
        if(plans[pos] == BCNEW):
            nplans[pos] = BCCOPY
        elif(plans[pos] == BCEXIST):
            nplans[pos] = BCSOURCE
    return nplans

def applyPlan(seq,plan,newvalue=None,copyvalue=NOVAL,existvalue=NOVAL,ensurevalue=NOVAL,sourcevalue=NOVAL):#{{{
    elempos = 0
    nseq = []
    for planelem in plan:
        if(planelem == BCNEW):
            nseq.append(newvalue)
        elif(planelem == BCEXIST):
            if(existvalue is NOVAL):
                nseq.append(seq[elempos])
            else:
                nseq.append(existvalue)
            elempos += 1
        elif(planelem == BCCOPY):
            if(copyvalue is NOVAL):
                nseq.append(seq[elempos])
            else:
                nseq.append(copyvalue)
            elempos += 1
        elif(planelem == BCSOURCE):
            if(sourcevalue is NOVAL):
                nseq.append(seq[elempos])
            else:
                nseq.append(sourcevalue)
            elempos += 1
        elif(planelem == BCENSURE):
            if(copyvalue is NOVAL):
                nseq.append(seq[elempos])
            else:
                nseq.append(ensurevalue)
            elempos += 1
        else:
            raise RuntimeError, "Unknown plan type"
    nseq.extend(seq[elempos:])
    return nseq#}}}



def flatFirstDims(array,ndim):#{{{
    """Flattens first ndim dims in numpy array into next dim"""
    oshape = array.shape
    ndim = ndim + 1
    rshape = (int(numpy.multiply.reduce(array.shape[:ndim])),) + array.shape[ndim:]
    return array.reshape(rshape)#}}}

def createDimSet(sourcepaths):#{{{
    """Create set of all dims in sourcepaths"""
    dimset = set()
    for dimpath in sourcepaths:
        dimset.update(dimpath)
    return dimset#}}}

def createDimParentDict(sourcepaths):#{{{
    """Create dictionary containing for each dim in sourcepaths 
    the previous occuring dims  in a  list"""
    parents = defaultdict(list)
    for dimpath in sourcepaths:
        parents[dimpath[0]].append(None)
        for pos in xrange(1,len(dimpath)):
            parents[dimpath[pos]].append(dimpath[pos-1])
    return parents#}}}

def extendParentDim(path, sourcepaths, length=1):#{{{
    length -= len(path.strip())   
    if(length <= 0):
        return path
    if path[0] is root:
        raise RuntimeError, "Could not get long enough dim path"

    ndims = []
    xpaths = set()
    for spath in sourcepaths:
        lastposs = spath.matchDimPath(path)
        for lastpos in lastposs:
            r = spath[:(lastpos + 1 - len(path))]
            xpaths.add(r)
        
    while(len(ndims) < length):
        xdims = set()
        nxpaths = set()
        for xpath in xpaths:
           if(not xpath):
               continue
           xdims.add(xpath[-1])
           nxpaths.add(xpath[:-1])
        
            
        if(not xdims or len(xdims) > 1):
            raise RuntimeError, "Cannot find unique parent for dim: " + str(path[0])
        xpaths = nxpaths
        ndims.append(xdims.pop())

    path = DimPath(*(ndims[::-1])) + path
    return path
    #}}}

def getArrayDimPathFromType(rtype):
    if(rtype.__class__ is rtypes.TypeArray):
        return rtype.dims + getArrayDimPathFromType(rtype.getSubType())
    else:
        return DimPath()

def getNestedArraySubType(rtype,depth):
    if depth == 0:
        return rtype
    if(rtype.__class__ is rtypes.TypeArray):
        depth -= len(rtype.dims)
        if depth < 0:
            return rtype
        else:
            return getNestedArraySubType(rtype.getSubType(),depth)
    else:
        return rtype

def dimsToArrays(dims, subtype):
    dims = dims[::-1]
    subtype = rtypes.TypeArray(dims=dims[:1],subtypes=(subtype,))

    for pos in xrange(1,len(dims)):
        subtype = rtypes.TypeArray(dims=dims[pos:(pos+1)],subtypes=(subtype,))
    return subtype

class DimPathRoot(Dim):
    pass
root = DimPathRoot(UNDEFINED, name="_root_")

class DimPathEnd(Dim):
    pass
end = DimPathEnd(UNDEFINED, name="_end_")

def identifyUniqueDimPathSource(source,dim_selector):#{{{
    res = identifyDimPathSource(source,dim_selector)
    if(len(res) > 1):
        raise RuntimeError, "Cannot choose between dim paths: " + str(res)
    elif(len(res) == 0):
        raise RuntimeError, "No matching dimpath found for selector: " + str(dim_selector)
    return res.pop()#}}}

def identifyDimPathSource(source,dim_selector):#{{{
    if(isinstance(dim_selector, context.Context)):
        dim_selector = context._apply(dim_selector, source)
    elif(isinstance(dim_selector, basestring)):
        if(dim_selector in [s.name for s in source._slices] or dim_selector[0].isupper()):
            dim_selector = getattr(source,dim_selector)

    if(isinstance(dim_selector, representor.Representor)):
        return set([s.dims for s in dim_selector._slices])

    return identifyDimPath(set([s.dims for s in source._slices]),dim_selector)#}}}

def identifyUniqueDimPath(source,dim_selector):#{{{
    res = identifyDimPath(source,dim_selector)
    if(len(res) > 1):
        raise RuntimeError, "Cannot choose between dim paths: " + str(res)
    elif(len(res) == 0):
        raise RuntimeError, "No matching dimpath found for selector: " + str(dim_selector)
    return res.pop()#}}}

def identifyDimPath(sourcepaths, dim_selector):#{{{
   
    if(isinstance(dim_selector, int)):
        udpath = uniqueDimPath(sourcepaths,only_unique=False)
        try: 
            dim_selector = udpath[dim_selector]
        except IndexError:
            raise RuntimeError, "Dimension at depth " + str(dim_selector) + " does not exist"

    elif(isinstance(dim_selector, long)):
        udpath = uniqueDimPath(sourcepaths,only_unique=False)
        if(dim_selector < 0):
            dim_selector += len(udpath)
        if not (dim_selector >= 0 and dim_selector < len(udpath)):
            return set()

        res = set()
        for spath in sourcepaths:
            if(len(spath.dims) > dim_selector):
                res.add((root,) + spath.dims[:(dim_selector + 1)])
        return res
    
    if(isinstance(dim_selector, Dim)):
        #fixme: root/end
        return set([DimPath(dim_selector)])
    elif(dim_selector is None):
        return set([commonDimPath(sourcepaths)])
    elif(isinstance(dim_selector, basestring)):
        return identifyDimPathParse(sourcepaths, dim_selector)
    elif(isinstance(dim_selector, DimPath)):
        #fixme: root/end
        return set([dim_selector])
    elif(isinstance(dim_selector, set)):
        res = set()
        for elem in dim_selector:
            res.update(identifyDimPath(sourcepaths, elem))
        return res 
    else:
        raise RuntimeError, "Unexpected dim selector: " + str(dim_selector)
    return res#}}}

def identifyDimPathParse(sourcepaths, dim_selector):#{{{
    dim_sel = []
    if(dim_selector[0] == "^"):
        dim_selector = dim_selector[1:]
        dim_sel.append(root)

    if(dim_selector and dim_selector[-1] == "$"):
        dim_selector = dim_selector[:-1]
        fixed_end = True
    else:
        fixed_end=False

    if(dim_selector):
        dim_sel.extend(dim_selector.split(":"))
    
    if(fixed_end):
        dim_sel.append(end)

    res = set()
    for spath in sourcepaths:
        res.update(identifyDimPathParseHelper(spath, dim_sel))
    return res#}}}

def identifyDimPathParseHelper(spath, dim_sel, outer=True):#{{{
    res = set()
    ds = dim_sel[0]
    
    if(ds is root):
        assert outer is True, "Root dim selection after begin"
        if(not dim_sel[1:]):
            res.add((spath[0],))
        else:
            _processRest(spath, dim_sel[1:], res, (root,))
    elif(ds is end):
        assert len(dim_sel) == 1, "Dim selectors found after end selector"
        res.add(tuple(spath) + (end,))
    elif(ds == "?"):
        if outer:
            for pos in xrange(len(spath)-1):
                _processRest(spath[(pos+1):], dim_sel[1:], res, (spath[pos],))
        elif(spath):
            _processRest(spath[1:], dim_sel[1:], res, (spath[0],))
    elif(ds == "*"):
        for pos in xrange(len(spath)):
            _processRest(spath[pos:], dim_sel[1:], res, spath[:pos])

    elif(ds == "+"):
        for pos in xrange(1,len(spath)):
            _processRest(spath[pos:], dim_sel[1:], res, spath[:pos])
    else:
        if outer:
            for pos in xrange(len(spath)):
                if(spath[pos].name == ds):
                    _processRest(spath[(pos+1):], dim_sel[1:], res, (spath[pos],))
        else:    
            if(spath and spath[0].name == ds):
                _processRest(spath[1:], dim_sel[1:], res, (spath[0],))
        
    return res        #}}}

def _processRest(spath, dim_selector, res, pathprefix):#{{{
    if(not dim_selector):
        if(pathprefix):
            res.add(DimPath(*pathprefix))
        return

    pathsuffixes = identifyDimPathParseHelper(spath, dim_selector,outer=False)
    for s in pathsuffixes:
        res.add(DimPath(*(pathprefix + s)))
        return#}}}


#}}}
