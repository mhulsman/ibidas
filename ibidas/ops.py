import copy
from collections import defaultdict
from itertools import izip_longest

from constants import *
from utils import util
from itypes import rtypes,dimpaths
from ops import *
from query_graph import Node

_delay_import_(globals(),"itypes","dimensions","typeops","convertors","casts")
_delay_import_(globals(),"itypes.type_attribute_freeze","freeze_protocol")

class Op(Node):
    __slots__ = ["link"]

    def assignLink(self,link):
        if(not hasattr(self, "link")):
            self.link = link

#pylint: disable-msg=E1101
class UnaryOp(Op):#{{{
    __slots__ = []
    """A unary operation is not only a description of an operation, but
    also the description of the output of that operation. As such, each unary
    op has a name, a type and a dimension attribute, as well as bookmarks.
    """

    __slots__ = ['name', 'type', 'dims','bookmarks']


    def __init__(self, name, rtype = rtypes.unknown, dims=dimpaths.DimPath(), bookmarks=set()):
        """Creates a slice object.

        :param name: name of slice (string)
        :param type: type of slice, optional, default = unknown
        :param dims: tuple of Dim objects, optional, default = ()
        """
        assert isinstance(name, basestring), "Name of slice should be a string"
        assert (name.lower() == name), "Name should be in lowercase"
        assert isinstance(rtype, rtypes.TypeUnknown), "Invalid type given"
        assert isinstance(dims, dimpaths.DimPath), "Dimensions of a slice should be a DimPath"
        assert isinstance(bookmarks,set), "Bookmarks should be a set"
        assert all([isinstance(bm, basestring) for bm in bookmarks]),"Bookmarks should be a string"

        self.name = name
        self.type = rtype
        self.dims = dims
        self.bookmarks = bookmarks
    
    def __repr__(self):
        res = self.name
        res += "="
        if(self.dims):
            dimstr = []
            for pos, dim in enumerate(self.dims):
                dimstr.append("[" + str(dim) + "]")
            res += "<".join(dimstr) + "<"
                
        res += str(self.type)
        return res#}}}

class ExtendOp(UnaryOp):#{{{
    __slots__ = []
#}}}

class DataOp(UnaryOp):#{{{
    __slots__ = ['data']
    def __init__(self, data, name=None, rtype=None, dims=dimpaths.DimPath(), bookmarks=set()):
        self.data = data
        UnaryOp.__init__(self, name, rtype, dims, bookmarks)#}}}

class UnaryUnaryOp(UnaryOp):#{{{
    __slots__ = ["source"]
    def __init__(self, source, name=None, rtype=None, dims=None, bookmarks=None):
        if(name is None):
            name = source.name
        if(rtype is None):
            rtype = source.type
        if(dims is None):
            dims = source.dims
        if(bookmarks is None):
            bookmarks = source.bookmarks
        self.source = source
        assert isinstance(source,Op),"Source of UnaryUnaryOpOp should be a slice"
        UnaryOp.__init__(self, name, rtype, dims, bookmarks)#}}}

class ChangeBookmarkOp(UnaryUnaryOp):#{{{
    __slots__ = []

    def __init__(self,source,add_bookmark=None,update_auto_bookmarks=None):
        nbookmarks = source.bookmarks.copy()

        if(not update_auto_bookmarks is None):
            for bm in list(nbookmarks):
                if(bm.startswith('!')):
                    nbookmarks.discard(bm)
                    nbookmarks.add("!" + update_auto_bookmarks + bm[1:])

        if(not add_bookmark is None):
            nbookmarks.add(add_bookmark)
        
        UnaryUnaryOp.__init__(self, source, bookmarks=nbookmarks)#}}}

class ChangeNameOp(UnaryUnaryOp):#{{{
    __slots__ = []
    def __init__(self,source, new_name):
        UnaryUnaryOp.__init__(self, source, name=new_name)#}}}

class ChangeDimOp(UnaryUnaryOp):#{{{
    __slots__ = []
    def __init__(self,source, pos, ndim):
        ndims,ntype = source.dims.updateDim(pos, ndim, source.type)
        UnaryUnaryOp.__init__(self, source, rtype=ntype,dims=ndims)#}}}

class ChangeDimPathOp(UnaryUnaryOp):#{{{
    __slots__ = []
    def __init__(self,source, ndims):
        UnaryUnaryOp.__init__(self, source, dims=ndims)#}}}

class CastOp(UnaryUnaryOp):#{{{
    __slots__ = ["cast_name"]
    def __init__(self, source, new_type):
        if(isinstance(new_type, rtypes.TypeUnknown)):
            cast_name = casts.canCast(source.type,new_type, len(source.dims))
        else:
            new_type,cast_name = casts.canCast(source.type,new_type, len(source.dims))
        assert not cast_name is False, "Cannot cast " + str(source.type) + " to " + str(new_type)
        self.cast_name = cast_name
        UnaryUnaryOp.__init__(self, source, rtype=new_type)#}}}
        
class DetectTypeOp(UnaryUnaryOp):#{{{
    __slots__ = []#}}}


class DetectFixedShapesOp(UnaryUnaryOp):
    __slots__ = []


class UnpackArrayOp(UnaryUnaryOp):#{{{
    """An slice which is the result of unpacking a source slice."""
    __slots__ = ["unpack_dims"]

    def __init__(self,slice,ndim=None):
        """Creates a new slice, and sets attributes.

        :param slice: Source slice to be unpacked"""
        stype = slice.type
        if(isinstance(stype, (rtypes.TypeSet, rtypes.TypeString))):
            slice = CastOp(slice,rtypes.TypeArray)
            stype = slice.type
            

        if(ndim is None):
            if(not stype.dims):
                ndim = 1
            else:
                ndim = len(stype.dims)
        
        unpack_dims = []
        rest_dims = []
        while ndim:
            assert isinstance(stype, rtypes.TypeArray), "Cannot unpack slice deep enough " + \
                                str(slice.name) + " as there is no array type"
            
            if(not stype.dims):
                nudims = (dimensions.Dim(UNDEFINED, variable=(True,) * (len(slice.dims) + len(unpack_dims)), 
                            has_missing=stype.has_missing),)
            else: 
                nudims = stype.dims[:ndim]
                rest_dims.extend(stype.dims[ndim:])
            last_type = stype
            stype = stype.subtypes[0]
            ndim -= len(nudims)
            unpack_dims.extend(nudims)
            if(ndim and isinstance(stype, (rtypes.TypeSet, rtypes.TypeString))):
                slice = UnpackArrayOp(slice,len(unpack_dims))
                slice = CastOp(slice, rtypes.TypeArray)
                unpack_dims = []
                stype = slice.type
                res_dims = []
        
        if(rest_dims):
            ntype = rtypes.TypeArray(last_type.has_missing, dimpaths.DimPath(*rest_dims), last_type.subtypes)
        else:
            ntype = stype

        self.unpack_dims = dimpaths.DimPath(*unpack_dims)
        UnaryUnaryOp.__init__(self, slice, rtype=ntype, dims=slice.dims + self.unpack_dims)#}}}

class InsertDimOp(UnaryUnaryOp):#{{{
    __slots__ = ["matchpoint","newdim"]
    def __init__(self,slice,matchpoint,ndim):
        assert len(slice.dims) >= matchpoint, "Matchpoint for dim insertion outside dimpath"
        assert ndim.shape == 1, "Length of inserted dim should be equal to 1"
        
        ndims,ntype = slice.dims.insertDim(matchpoint,ndim, slice.type)
        self.matchpoint = matchpoint
        self.newdim = ndim
        UnaryUnaryOp.__init__(self,slice,rtype=ntype,dims=ndims)        #}}}

class EnsureCommonDimOp(UnaryUnaryOp):#{{{
    __slots__ = ["refslices","checkpos"]
    def __init__(self,slice,refslices,checkpos,bcdim):
        self.checkpos = checkpos
        self.refslices = refslices
        ndims,ntype = slice.dims.updateDim(checkpos, bcdim, slice.type)
        UnaryUnaryOp.__init__(self, slice, rtype=ntype,dims=ndims)#}}}

class BroadcastOp(UnaryUnaryOp):#{{{
    __slots__ = ["refsliceslist","plan","bcdims","partial"]
    
    def __init__(self,slice,refslices,plan,bcdims,partial=False):
        self.refsliceslist = refslices
        self.plan = plan
        self.bcdims = bcdims
        self.partial=partial

        ndims = slice.dims
        ntype = slice.type
        bcpos = 0
        for pos, planelem in enumerate(plan):
            if(planelem == BCEXIST):
                assert refslices[bcpos],"No refslices for broadcast of dim: " + str(bcdims[pos])
                bcpos += 1
                ndims, ntype = ndims.updateDim(pos, bcdims[pos], ntype)
            elif(planelem == BCCOPY or planelem == BCSOURCE):
                pass
            else:
                raise RuntimeError, "Unknown plan element in plan: " + str(plan)
            
        UnaryUnaryOp.__init__(self, slice, dims=ndims,rtype=ntype)#}}}

def broadcast(slices,mode="pos", partial=False):#{{{
    if not slices:
        return ((),())
    slicedimpaths = [s.dims for s in slices]
    if(mode == "dim"):
        bcdims, bcplan = dimpaths.planBroadcastMatchDim(slicedimpaths)
    elif(mode == "pos"):
        bcdims, bcplan = dimpaths.planBroadcastMatchPos(slicedimpaths)
    else:
        raise RuntimeError, "Unknown broadcast mode: " + str(mode)

    references = defaultdict(list)
    for bcdim in bcdims:
        for slice in slices:
            if bcdim in slice.dims:
                references[bcdim].append(slice)

    nslices = []
    for plan,slice in zip(bcplan,slices):
        slice = apply_broadcast_plan(slice, plan, bcdims, references, partial)
        nslices.append(slice)
    return (nslices, bcplan)
    #}}}

def broadcast_fromplan(slice, refslices, plan, origdims, bcdims,  partial=False):#{{{
    references = defaultdict(list)
    for bcdim in bcdims:
        for rslice in refslices:
            if bcdim in rslice.dims:
                references[bcdim].append(rslice)

    sbcdims,splan = dimpaths.planBroadcastFromPlan(slice.dims, plan, origdims, bcdims)
    slice = apply_broadcast_plan(slice, splan, sbcdims, references, partial)
    return (slice, splan)
    #}}}

def apply_broadcast_plan(slice,plan, bcdims, references, partial):#{{{
    nplan = []
    active_bcdims = []
    for dimpos, bcdim, planelem in zip(range(len(bcdims)),bcdims, plan):
        if(planelem == BCNEW):
            ndim = dimensions.Dim(1)
            slice = InsertDimOp(slice,dimpos,ndim)
            active_bcdims.append(bcdim)
            nplan.append(BCEXIST)
        elif(planelem == BCENSURE):
            slice = EnsureCommonDimOp(slice,references[bcdim],dimpos,bcdim)
            nplan.append(BCSOURCE)
        elif(planelem == BCEXIST):
            active_bcdims.append(bcdim)
            nplan.append(planelem)
        else:
            nplan.append(planelem)
    if(active_bcdims):                
        slice = BroadcastOp(slice,[references[bcdim] for bcdim in active_bcdims], nplan, bcdims, partial)
    return slice
#}}}

def broadcastParentsFromPlan(slice, seldimpath, plan, origdims, bcdims, refslices, partial):#{{{
    used_dims = [False,] * len(slice.dims)
    if not plan:
        return slice
    while True:
        #determine filter dim
        lastpos = slice.dims.matchDimPath(seldimpath)
        for filterpos in lastpos[::-1]:
             if(not used_dims[filterpos]):
                break
        else:
            break

        #pack up to and including filter dim
        packdepth = len(slice.dims) - filterpos
        if(packdepth):
            slice = PackArrayOp(slice,packdepth)

        #broadcast to constraint
        slice,splan = broadcast_fromplan(slice, refslices, plan, origdims, bcdims, partial)

        #adapt used_dims
        used_dims = dimpaths.applyPlan(used_dims,splan,newvalue=True,copyvalue=True,ensurevalue=True,existvalue=True)
        used_dims[filterpos] = True

        #unpack dims
        if(packdepth):
            slice = UnpackArrayOp(slice,packdepth)
            
    return slice#}}}

class PermuteDimsOp(UnaryUnaryOp):#{{{
    __slots__ = ["permute_idxs"]
    
    def __init__(self,slice,permute_idxs):
        assert len(permute_idxs) == len(slice.dims), "Number of permute indexes not equal to number of dimensions"
        
        ndims,ntype = slice.dims.permuteDims(permute_idxs,slice.type)
        self.permute_idxs = permute_idxs
        UnaryUnaryOp.__init__(self,slice, dims=ndims,rtype=ntype)#}}}

class SplitDimOp(UnaryUnaryOp):#{{{
    __slots__ = ["pos","lshape","rshape"]

    def __init__(self,slice, pos,lshape,rshape,ldim,rdim):
        npath,ntype = slice.dims.updateDim(pos, (ldim,rdim),slice.type)
        self.pos = pos
        self.lshape = lshape
        self.rshape = rshape

        UnaryUnaryOp.__init__(self,slice,dims=npath,rtype=ntype)#}}}

class ShapeOp(UnaryUnaryOp):#{{{
    __slots__ = ["pos"]
    def __init__(self,slice,pos):
        d = slice.dims[pos]
        ntype = rtypes.unknown
        self.pos = pos
        UnaryUnaryOp.__init__(self,slice, name=d.name,rtype=ntype,dims=dimpaths.DimPath())#}}}

class FilterOp(UnaryUnaryOp):#{{{
    __slots__ = ["constraint","has_missing"]

    def __init__(self,slice,constraint, ndim,refconstraint=None):
        if refconstraint is None:
            refconstraint = constraint

        stype = slice.type
        assert isinstance(stype, rtypes.TypeArray), "Filter on non-array type not possible"
        
        if(isinstance(constraint.type,rtypes.TypeArray)):
            has_missing = constraint.type.subtypes[0].has_missing
        else:
            has_missing = constraint.type.has_missing
        self.has_missing = has_missing            

        sdims = stype.dims
        if(ndim is None):
            sdims, subtype = sdims.removeDim(0, refconstraint, stype.subtypes[0])
        else:
            sdims, subtype = sdims.updateDim(0, ndim, stype.subtypes[0])
        
        
        if(sdims):
            ntype = dimpaths.dimsToArrays(sdims, subtype)
        else:
            ntype = subtype
        
        if(has_missing):
            ntype = ntype.setHasMissing(True)
            if not subtype.hasMissingValInfo():
                slice = DetectFixedShapesOp(slice)
      
        self.constraint = constraint
        UnaryUnaryOp.__init__(self, slice, rtype=ntype)#}}}

def filter(slice,constraint,seldimpath, ndim, mode="dim"):#{{{
    used_dims = [False,] * len(slice.dims)
    while True:
        #determine filter dim
        lastpos = slice.dims.matchDimPath(seldimpath)
        for filterpos in lastpos[::-1]:
             if(not used_dims[filterpos]):
                break
        else:
            break

        #pack up to and including filter dim
        packdepth = len(slice.dims) - filterpos
        if(packdepth):
            slice = PackArrayOp(slice,packdepth)


        #prepare adaptation of ndim.dependent
        if(not ndim is None):
            dep = list(ndim.dependent)
            while(len(dep) < len(constraint.dims)):
                dep.insert(0,False)

        #broadcast to constraint
        (slice,tconstraint),(splan,cplan) = broadcast([slice,constraint],mode=mode, partial=True)

        #adapt ndim to braodcast, apply filter
        #FIXME: change dependent dims of slice.type

        if(not ndim is None):
            if(isinstance(tconstraint.type,rtypes.TypeSlice)):
                ndep = dimpaths.applyPlan(dep[::-1],cplan,newvalue=True, copyvalue=True, ensurevalue=True)
            else:
                ndep = dimpaths.applyPlan(dep[::-1],cplan,newvalue=False)
            xndim = ndim.changeDependent(tuple(ndep[::-1]), slice.dims)
        else:
            xndim = ndim
        slice = FilterOp(slice,tconstraint,xndim,constraint)

        #adapt used_dims
        used_dims = dimpaths.applyPlan(used_dims,splan,newvalue=True,copyvalue=True,ensurevalue=True, existvalue=True)
        
        #handle dim removal, filter_dim is used
        if(ndim is None): #dimension removed/collapsed
            packdepth -= 1
            del used_dims[len(cplan)]
        else:
            used_dims[len(cplan)] = True

        #unpack dims
        if(packdepth):
            slice = UnpackArrayOp(slice,packdepth)
            
    return slice#}}}

class FlatAllOp(UnaryUnaryOp):#{{{
    __slots__ = []

    def __init__(self,slice, ndim):
        stype = slice.type
        sdims = slice.dims
        for i in xrange(len(sdims)):
            stype = stype._removeDepDim((i+1), None)
        stype = stype._insertDepDim(1,ndim)
        dims = dimpaths.DimPath(ndim)

        UnaryUnaryOp.__init__(self, slice, rtype=stype,dims=dims)#}}}

class FlatDimOp(UnaryUnaryOp):#{{{
    __slots__ = ["flatpos"]

    def __init__(self,slice,flatpos, ndim):
        stype = slice.type
        sdims = slice.dims
        sdims,stype = sdims.removeDim(flatpos,None,stype)
        sdims,stype = sdims.updateDim(flatpos-1,ndim,stype)

        self.flatpos = flatpos
        UnaryUnaryOp.__init__(self, slice, rtype=stype,dims=sdims)#}}}

class UnpackTupleOp(UnaryUnaryOp):#{{{
    """A slice which is the result of unpacking a tuple slice."""
    __slots__ = ["tuple_idx"]
    
    def __init__(self, slice, idx):
        """Creates a new slice, using source `slice`, by 
        extracting the `idx` subtype.
        
        :param slice: new slice
        :param idx:   index of tuple attribute to be unpacked"""
        
        stype = slice.type
        assert isinstance(stype, rtypes.TypeTuple), "Cannot unpack slice " + \
                                str(slice.name) + " as it is not a tuple"
        
        assert 0 <= idx < len(slice.type.subtypes), \
            "Tuple index invalid, outside range of available attributes"

        ntype = slice.type.subtypes[idx]
        
        if(slice.type.fieldnames):
            name = slice.type.fieldnames[idx]
        else:
            name = "f" + str(idx)

        UnaryUnaryOp.__init__(self, slice, name=name, rtype=ntype)
        self.tuple_idx = idx#}}}

class PackArrayOp(UnaryUnaryOp):#{{{
    __slots__ = ["pack_dims"]

    def __init__(self, pslice, ndim=1):
        assert len(pslice.dims) >= ndim, "Op does not have enough dimensions to pack as " + str(ndim) + "-dimensional array"
        
        dims = pslice.dims[-ndim:]
        self.pack_dims = dims
        ntype = dimpaths.dimsToArrays(dims,pslice.type)
        UnaryUnaryOp.__init__(self, pslice, rtype=ntype, dims=pslice.dims[:-ndim])#}}}

class PackListOp(PackArrayOp):#{{{
    __slots__ = []

    def __init__(self, pslice):
        PackArrayOp.__init__(self, pslice, 1)
        #}}}

class ToPythonOp(UnaryUnaryOp):#{{{
    __slots__ = []

    def __init__(self, pslice):
        ntype = pslice.type.copy()
        UnaryUnaryOp.__init__(self, pslice, rtype=ntype)#}}}
        
class NoneToMissingOp(UnaryUnaryOp):#{{{
    __slots__ = []
    def __init__(self, slice):
        assert slice.type.has_missing, "Cannot apply none to missing on type without has_missing"
        UnaryUnaryOp.__init__(self, slice)#}}}
    
class FreezeOp(UnaryUnaryOp):#{{{
    __slots__ = []
#}}}

def ensure_frozen(slice):#{{{
    if(freeze_protocol.needFreeze(slice.type)):
        return FreezeOp(slice)
    else:
        return slice#}}}

class UnaryFuncOp(UnaryUnaryOp):#{{{
    __slots__ = ["funcname","sig", "kwargs"]
    def __init__(self, slice, funcname, sig, outparam, dims=None, **kwargs):
        self.funcname = funcname
        self.sig = sig
        self.kwargs = kwargs

        if(dims is None):
            dims = slice.dims
        UnaryUnaryOp.__init__(self, slice, name=outparam.name, rtype=outparam.type, dims=dims)#}}}

class UnaryFuncElemOp(UnaryFuncOp):#{{{
    __slots__ = []
    
    def __init__(self, funcname, sig, outparam, slice, **kwargs):
        UnaryFuncOp.__init__(self, slice, funcname, sig, outparam, **kwargs)
        #}}}

class UnaryFuncSeqOp(UnaryFuncOp):#{{{
    __slots__ = ["packdepth"]

    
    def __init__(self, funcname, sig, outparam, packdepth, slice, **kwargs):
        self.packdepth = packdepth
        UnaryFuncOp.__init__(self, slice, funcname, sig, outparam, **kwargs)
        #}}}

class UnaryFuncAggregateOp(UnaryFuncOp):#{{{
    __slots__ = ["packdepth"]
    
    def __init__(self, funcname, sig, outparam, packdepth, slice, **kwargs):
        self.packdepth = packdepth
        sdims, ntype = slice.dims.removeDim(len(slice.dims) - packdepth, (funcname,outparam), outparam.type)

        UnaryFuncOp.__init__(self, slice, funcname, sig, outparam, dims=sdims, **kwargs)
        self.type = ntype
        #}}}

class MultiUnaryOp(UnaryOp):#{{{
    __slots__ = ["sources"]
    def __init__(self, source_slices, name, rtype=rtypes.unknown, 
                            dims=dimpaths.DimPath(), bookmarks=set()):
        self.sources = tuple(source_slices)
        UnaryOp.__init__(self, name, rtype, dims, bookmarks)#}}}

class BinFuncOp(MultiUnaryOp):#{{{
    __slots__ = ["funcname","sig"]
    def __init__(self, lslice, rslice, funcname, sig, outparam, dims=None):
        self.funcname = funcname
        self.sig = sig

        if(dims is None):
            assert lslice.dims == rslice.dims, "Op dims not equal, and no dim given"
            dims = lslice.dims
        nbookmarks = lslice.bookmarks | rslice.bookmarks

        MultiUnaryOp.__init__(self, (lslice, rslice), name=outparam.name, rtype=outparam.type, dims=dims, bookmarks=nbookmarks)#}}}

class BinFuncElemOp(BinFuncOp):#{{{
    def __init__(self, funcname, sig, outparam, left, right):
        dims = left.dims
        assert all([d1 == d2 for d1, d2 in zip(dims, right.dims)]), \
                    "Dimensions of slices do not match"
        
        BinFuncOp.__init__(self, left, right, funcname, sig, outparam, dims)#}}}

class PackTupleOp(MultiUnaryOp):#{{{
    __slots__ = []
    ocls = rtypes.TypeTuple

    def __init__(self, slices, field="data"):
        cdim = set([slice.dims for slice in slices])
        assert len(cdim) == 1, "Packing tuple on slices with different dims"
        
        fieldnames = [slice.name for slice in slices]
        subtypes = [slice.type for slice in slices]
        ntype = self.ocls(False, tuple(subtypes), tuple(fieldnames))
        nbookmarks = reduce(set.union,[slice.bookmarks for slice in slices])
        MultiUnaryOp.__init__(self, slices, name=field, rtype=ntype, dims=iter(cdim).next(),bookmarks=nbookmarks)#}}}

class PackDictOp(PackTupleOp):#{{{
    __slots__ = ["with_missing"]
    ocls = rtypes.TypeRecordDict

    def __init__(self, slices, field="data", with_missing=False):
        self.with_missing = with_missing
        PackTupleOp.__init__(self, slices, field)#}}}


class PackIndexDictOp(MultiUnaryOp):#{{{
    __slots__ = []
    ocls = rtypes.TypeIndexDict

    def __init__(self, slices, field="data"):
        assert len(slices) == 2, "Index dict op should have two slices"
        assert slices[0].dims == slices[1].dims, "Key and value of dict should have same dimension"

        slices = [PackArrayOp(slice) for slice in slices]
        ndims = dimpaths.DimPath(dimensions.Dim(UNDEFINED, (), name="dict"))
        subtypes = [dimpaths.dimsToArrays(ndims, slice.type.subtypes[0]) for slice in slices]
        
        fieldnames = [slice.name for slice in slices]
        ntype = self.ocls(False, tuple(subtypes), tuple(fieldnames))
        nbookmarks = reduce(set.union,[slice.bookmarks for slice in slices])
        MultiUnaryOp.__init__(self, slices, name=field, rtype=ntype, dims=slices[0].dims, bookmarks=nbookmarks)#}}}
        #}}}


class TakeOp(MultiUnaryOp):
    __slots__ = ['allow_missing','keep_missing']
    def __init__(self, source_slice, take_slice, allow_missing=False, keep_missing=False):
        assert isinstance(source_slice.type, rtypes.TypeIndexDict), "Take left op should have type index dict"
        source_slice, take_slice = broadcast((source_slice, take_slice), mode="dim")[0]
        ntype = source_slice.type.subtypes[1].subtypes[0]
        if allow_missing:
            ntype = ntype.setHasMissing(True)
        elif keep_missing:
            ntype = casts.castImplicitCommonType(ntype,take_slice.type)

        if keep_missing:
            name = take_slice.name
        else:
            name = source_slice.type.fieldnames[1]
        nbookmarks = source_slice.bookmarks | take_slice.bookmarks
        self.allow_missing = allow_missing
        self.keep_missing = keep_missing
        MultiUnaryOp.__init__(self, [source_slice, take_slice], name=name, rtype=ntype, dims=take_slice.dims, bookmarks = nbookmarks)#}}}

class GroupIndexOp(MultiUnaryOp):#{{{
    def __init__(self, slices):
        assert len(set([slice.dims for slice in slices])) == 1, "Group index slices should have same dimension"
        assert all([isinstance(slice.type,rtypes.TypeArray) for slice in slices]), "Group index slices should be arrays"
        ndims = slices[0].dims
         
        newdims = []
        dep = (True,) * len(ndims)
        for pos, slice in enumerate(slices):
            xdep = (False,) * pos + dep
            newdims.append(dimensions.Dim(UNDEFINED,dependent=xdep,name="g" + slice.name))

        sdim = dimensions.Dim(UNDEFINED,(True,) * (len(ndims) + len(newdims)),name="g" + slices[0].type.dims[0].name)
        stype = rtypes.TypeInt64()
        rtype = dimpaths.dimsToArrays(dimpaths.DimPath(sdim), stype)
        rtype = dimpaths.dimsToArrays(dimpaths.DimPath(*newdims), rtype)
        MultiUnaryOp.__init__(self,slices, name="groupindex", rtype=rtype, dims=ndims)#}}}

class HArrayOp(MultiUnaryOp):#{{{
    __slots__ = []

    def __init__(self, slices, name=None):
        cdim = set([slice.dims for slice in slices])
        assert len(cdim) == 1, "Packing tuple on slices with different dims"
        
        subtypes = [slice.type for slice in slices]
        assert len(set(subtypes)) == 1, "HArray can only be applied if types are equal"

        ndim = dimensions.Dim(len(slices), name=name)
        ntype = rtypes.TypeArray(False, dimpaths.DimPath(ndim), (subtypes[0],))

        nbookmarks = reduce(set.union,[slice.bookmarks for slice in slices])

        slicename = "_".join([slice.name for slice in slices])
        MultiUnaryOp.__init__(self, slices, name=slicename, rtype=ntype, dims=iter(cdim).next(),bookmarks=nbookmarks)#}}}

class FixateOp(MultiUnaryOp):
    __slots__ = []
    def __init__(self, slices):
        MultiUnaryOp.__init__(self, slices, "result")

class GatherOp(FixateOp):
    __slots__ = []

class MultiOp(Op):
    __slots__ = ["results"]
    
    def __init__(self, results):
        assert isinstance(results, tuple), "Results should be a tuple"
        self.results = results

class SelectOp(UnaryUnaryOp):#{{{
    __slots__ = ["index"]
    def __init__(self, slice, index, name, rtype, ndims, bookmarks):
        assert isinstance(slice, MultiOp), "Source lice of SelectOp should be a multiop"
        self.index = index
        UnaryUnaryOp.__init__(self, slice, name=name, rtype=rtype, dims=ndims, bookmarks=bookmarks)#}}}

class UnaryMultiOp(MultiOp):
    __slots__ = ["source"]

class MultiMultiOp(MultiOp):#{{{
    __slots__ = ["sources"]

    def __init__(self, sources, results):
        assert isinstance(sources, tuple), "Sources should be a tuple"
        self.sources = sources
        MultiOp.__init__(self, results)#}}}


class EquiJoinIndexOp(MultiMultiOp):#{{{
    __slots__ = ["jointype"]

    def __init__(self, leftslice, rightslice, jointype="inner", mode="dim"):
        assert leftslice.dims == rightslice.dims, "Parent dims of join fields should be equal"
        if leftslice.type.dims[0].name == rightslice.type.dims[0].name:
            name= leftslice.type.dims[0].name
        else:
            name= leftslice.type.dims[0].name + "_" + rightslice.type.dims[0].name
        ndim = dimensions.Dim(UNDEFINED, (True,) * len(leftslice.dims), name=name)

        ltype = rtypes.TypePlatformInt(has_missing=(jointype == "right" or jointype=="full"))
        ltype = rtypes.TypeArray(subtypes=(ltype,), dims=dimpaths.DimPath(ndim))
        
        rtype = rtypes.TypePlatformInt(has_missing=(jointype == "left" or jointype=="full"))
        rtype = rtypes.TypeArray(subtypes=(rtype,), dims=dimpaths.DimPath(ndim))

        r1 = SelectOp(self,0,leftslice.name, ltype,leftslice.dims, leftslice.bookmarks)
        r2 = SelectOp(self,1,rightslice.name, rtype,leftslice.dims, rightslice.bookmarks)
        self.jointype = jointype
        MultiMultiOp.__init__(self, (leftslice,rightslice),(r1,r2))#}}}




