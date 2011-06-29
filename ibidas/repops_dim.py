from constants import *
import repops
from itertools import izip_longest

_delay_import_(globals(),"itypes","rtypes","dimpaths","dimensions")
_delay_import_(globals(),"ops")
_delay_import_(globals(),"repops_slice")
_delay_import_(globals(),"utils","util")

class UnpackArray(repops.UnaryOpRep):
    def _process(self,source, name = None, ndim=None):
        """Operation to unpack array typed slices

        :param source: source with active slices which should be unpacked
        :param name: (Optional) name of dimension to unpack. If not given,  unpack all.
        """
        if not source._typesKnown():
            return

        nslices = []                       #new active slices
        for slice in source._slices:
            #if name param, but does not match
            if(isinstance(slice.type,rtypes.TypeArray)):
                if(not name is None):
                    dimindex = slice.type.dims.getDimIndexByName(name)
                    if(not dimindex is None):
                        slice = ops.UnpackArrayOp(slice,ndim=dimindex)
                else:
                    slice = ops.UnpackArrayOp(slice,ndim=ndim)
            nslices.append(slice)

        return self._initialize(tuple(nslices))

class DimRename(repops.UnaryOpRep):#{{{
    def _sprocess(self, source, *names, **kwds):
        if(names):
            unique_dimpath = util.unique(sum([slice.dims for slice in source._slices],dimpaths.DimPath())) 
            assert (len(names) <= len(unique_dimpath)),\
                    "Number of new dim names larger than number of dims"
            
            name_dict = {}
            for newname,dim  in zip(names,unique_dimpath):
                name_dict[dim] = newname
            kwds.update(name_dict)
        
        nslices = [ops.ChangeDimPathOp(slice, slice.dims.changeNames(kwds))
                                                    for slice in source._slices]
        return self._initialize(tuple(nslices))#}}}



class Redim(repops.UnaryOpRep):
    def _sprocess(self, source, *args, **kwds):
        dimname = args[0]
        dimsel = []
        slicesel = []
        for arg in args[1:]:
            dimsel.append(0)
            slicesel.append(arg)
        for k,v in kwds.iteritems():
            dimsel.append(v)
            slicesel.append(k)
        
        selslices = repops_slice.Project(source, *slicesel)._slices
       
        nslices = list(source._slices)
        dims = set()
        for slice,dim in zip(selslices,dimsel):
            slice = nslices[nslices.index(slice)]
            dimidxs = slice.dims.getDimIndices(dim)
            for dimidx in dimidxs:
                dims.add(slice.dims[dimidx])

        ndim = dimensions.toCommonDim(dimname, dims)
        for slice,dim in zip(selslices,dimsel):
            sliceidx = nslices.index(slice)
            slice = nslices[sliceidx]
            dimidxs = slice.dims.getDimIndices(dim)
            for dimidx in dimidxs:
                slice = ops.ChangeDimOp(slice, dimidx, ndim)
            nslices[sliceidx] = slice                            
       
        return self._initialize(tuple(nslices))

class Shape(repops.UnaryOpRep):
    def _sprocess(self,source):
        dimnames = set()
        nslices = []
        for slice in source._slices:
            for pos, dim in enumerate(slice.dims):
                if dim.name in dimnames:
                    continue
                nslice = ops.ShapeOp(slice,pos)
                nslices.append(nslice)
                dimnames.add(dim.name)
        return self._initialize(tuple(nslices))
       

class InsertDim(repops.UnaryOpRep):
    def _sprocess(self, source, insertpoint, name=None):
        assert len(dimpaths.uniqueDimPath([s.dims for s in source._slices],only_unique=False)) >= insertpoint, "No unique dimpath covering dim insertion point"
        ndim = dimensions.Dim(1,name=name)
        nslices = []
        for slice in source._slices:
            slice = ops.InsertDimOp(slice,insertpoint,ndim)
            nslices.append(slice)
        return self._initialize(tuple(nslices))

class SplitDim(repops.UnaryOpRep):
    def _sprocess(self,source,lshape,rshape,lname=None,rname=None,dimsel=None):
        selpath = dimpaths.identifyUniqueDimPathSource(source,dimsel)
        nslices = []

        if(isinstance(lshape,int)):
            ldim = dimensions.Dim(lshape,name=lname)
        else:
            ldim = dimensions.Dim(UNDEFINED,dependent=(True,) * len(selpath.strip()), name=lname)

        if(isinstance(rshape,int)):
            rdim = dimensions.Dim(rshape,name=rname)
        else:
            rdim = dimensions.Dim(UNDEFINED,dependent=(True,) * (len(selpath.strip()) + 1), name=rname)
        
        for slice in source._slices:
            lastposs = slice.dims.matchDimPath(selpath)
            for lastpos in lastposs:
                slice = ops.SplitDimOp(slice,lastpos,lshape,rshape,ldim,rdim)
            nslices.append(slice)
            
        return self._initialize(tuple(nslices))


class PermuteDims(repops.UnaryOpRep):
    def _sprocess(self, source, permute_idxs):
        cpath = dimpaths.commonDimPath([s.dims for s in source._slices])
        if len(permute_idxs) > len(cpath):
            raise RuntimeError, "Permute index length longer than common dimensions"
        assert len(permute_idxs) == len(set(permute_idxs)), "Permute indices not unique"
        assert min(permute_idxs) == 0, "Lowest permute index should be 0"
        assert max(permute_idxs) == len(permute_idxs) - 1, "Highest permute index should be equal to: " + str(len(permute_idxs)-1)

        nslices = []
        for slice in source._slices:
            pidx = list(permute_idxs) + range(max(permute_idxs) + 1, len(slice.dims))
            slice = ops.PermuteDimsOp(slice,pidx)
            nslices.append(slice)
        return self._initialize(tuple(nslices))


class Array(repops.UnaryOpRep):
    def _sprocess(self, source, tolevel=None):
        if tolevel is None:
            nslices = [ops.PackArrayOp(slice) for slice in source._slices]
        else:
            nslices = []
            for slice in source._slices:
                if(len(slice.dims) > tolevel):
                    nslices.append(ops.PackArrayOp(slice, len(slice.dims) - tolevel))
                else:
                    nslices.append(slice)

        return self._initialize(tuple(nslices))


class Level(repops.UnaryOpRep):
    def _sprocess(self, source, tolevel=1):
        nslices = []
        for slice in source._slices:
            if(len(slice.dims) > tolevel):
                nslices.append(ops.PackArrayOp(slice, len(slice.dims) - tolevel))
            else:
                nslices.append(slice)

        if nslices:
            upaths = dimpaths.uniqueDimPath([s.dims for s in nslices],only_unique=True)
            assert len(upaths) >= tolevel, "Level can only be executed on slices with similar root dimensions"
        
        nslices,plan = ops.broadcast(nslices, mode="dim")
        return self._initialize(tuple(nslices))


class FlatAll(repops.UnaryOpRep):
    def _sprocess(self,source,name=None):
        nslices = ops.broadcast(source._slices,mode="dim")[0]
        dims = nslices[0].dims
        
        shape = 1
        for dim in dims:
            if(dim.shape == UNDEFINED):
                shape = UNDEFINED
                break
            else:
                shape *= dim.shape
        if(name is None):
            name = "_".join([dim.name for dim in dims])

        ndim = dimensions.Dim(shape, name=name)
        
        nnslices = []
        for slice in nslices:
            nnslices.append(ops.FlatAllOp(slice, ndim))

        return self._initialize(tuple(nnslices))

class Flat(repops.UnaryOpRep):
    def _sprocess(self,source,name=None,dim=-1):
        selpath = dimpaths.identifyUniqueDimPathSource(source,dim)
        nslices = self._apply(source._slices, selpath, name)

        return self._initialize(tuple(nslices))

    @classmethod
    def _apply(cls, fslices, selpath, name=None):
        #create new merged dimension
        selpath = dimpaths.extendParentDim(selpath,[s.dims for s in fslices],max(2,1 + len(selpath[-1].dependent)))
        
        #determine new dim
        if(selpath[-1].shape != UNDEFINED and selpath[-2].shape != UNDEFINED):
            shape = selpath[-2].shape * selpath[-1].shape
        else:
            shape = UNDEFINED

        if(name is None):
            name = selpath[-2].name + "_" + selpath[-1].name

        dependent = tuple([left or right for left,right in 
                        izip_longest(selpath[-2].dependent, selpath[-1].dependent[1:],fillvalue=False)])

        ndim = dimensions.Dim(shape, dependent=dependent, has_missing = selpath[-1].has_missing or selpath[-2].has_missing, name=name)
        
        bcdim = dimensions.Dim(1)

        #find refslices
        refslices = [s for s in fslices if selpath[-1] in s.dims]

        #process slices
        nslices = []
        for slice in fslices:
            sdims = slice.dims
            lastpos = sdims.matchDimPath(selpath[:-1])
            while(lastpos):
                flatpos = lastpos[0] + 1
                if(len(sdims) <= flatpos or sdims[flatpos] != selpath[-1]):
                    slice = ops.InsertDimOp(slice,flatpos,bcdim)
                    bcdims = slice.dims[:flatpos] + (selpath[-1],) + slice.dims[flatpos:]
                    plan = [BCSOURCE] * len(slice.dims[:flatpos]) + [BCEXIST] + [BCSOURCE] * len(slice.dims[flatpos:])
                    slice = ops.BroadcastOp(slice,[refslices],plan,bcdims)
                slice = ops.FlatDimOp(slice,flatpos,ndim)
                if(len(lastpos) > 1):
                    sdims = slice.dims
                    lastpos = sdims.matchDimPath(selpath[:-1])
                else:
                    lastpos = []
            nslices.append(slice)
        return nslices

class GroupIndex(repops.UnaryOpRep):
    def _sprocess(self, source):
        nslices = [ops.ensure_frozen(slice) for slice in source._slices]
        nslices = ops.broadcast(nslices,mode="dim")[0]
        nslices = [ops.PackArrayOp(nslice,1) for nslice in nslices]

        nslice = ops.GroupIndexOp(nslices)
        nslice = ops.UnpackArrayOp(nslice, len(nslices))
        return self._initialize((nslice,))



