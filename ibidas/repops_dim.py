from constants import *
import repops
from itertools import izip_longest

_delay_import_(globals(),"itypes","rtypes","dimpaths","dimensions")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"utils","util")

class UnpackArray(repops.UnaryOpRep):
    def _process(self,source, name = None, ndim=None):
        """Operation to unpack array typed slices

        Parameters
        ----------
        source: source with active slices which should be unpacked
        name: (Optional) name of dimension to unpack. If not given,  unpack all.
        """
        if not source._state & RS_TYPES_KNOWN:
            return

        nslices = []                       #new active slices
        for slice in source._slices:
            #if name param, but does not match
            if(isinstance(slice.type,rtypes.TypeArray)):
                if(not name is None):
                    dimindex = slice.type.dims.getDimIndexByName(name)
                    if(not dimindex is None):
                        slice = slices.ensure_normal_or_frozen(slices.UnpackArraySlice(slice,ndim=dimindex))
                else:
                    slice = slices.ensure_normal_or_frozen(slices.UnpackArraySlice(slice,ndim=ndim))
            nslices.append(slice)

        return self._initialize(tuple(nslices),RS_CHECK)

class DimRename(repops.UnaryOpRep):#{{{
    def _process(self, source, *names, **kwds):
        if not source._state & RS_SLICES_KNOWN:
            return
        if(names):
            unique_dimpath = util.unique(sum([slice.dims for slice in source._slices],dimpaths.DimPath())) 
            assert (len(names) <= len(unique_dimpath)),\
                    "Number of new dim names larger than number of dims"
            
            name_dict = {}
            for newname,dim  in zip(names,unique_dimpath):
                name_dict[dim] = newname
            kwds.update(name_dict)
        
        nslices = [slices.ChangeDimPathSlice(slice, slice.dims.changeNames(kwds))
                                                    for slice in source._slices]
        return self._initialize(tuple(nslices),source._state)#}}}

class InsertDim(repops.UnaryOpRep):
    def _process(self, source, insertpoint, name=None):
        if not source._state & RS_SLICES_KNOWN:
            return
        
        assert len(dimpaths.uniqueDimPath([s.dims for s in source._slices],only_complete=False)) >= insertpoint, "No unique dimpath covering dim insertion point"
        ndim = dimensions.Dim(1,name=name)
        nslices = []
        for slice in source._slices:
            slice = slices.InsertDimSlice(slice,insertpoint,ndim)
            nslices.append(slice)
        return self._initialize(tuple(nslices),source._state)


@repops.delayable()
def rarray(source, dim=None, ndim=1):
    return repops.ApplyFuncRep(source, repops.apply_slice, slices.PackArraySlice, dim, ndim=1)

@repops.delayable()
def rlist(source, dim=None):
    return repops.ApplyFuncRep(source, repops.apply_slice, slices.PackListSlice, dim)

class FlatAll(repops.UnaryOpRep):
    def _process(self,source,name=None):
        if not source._state & RS_SLICES_KNOWN:
            return
        nslices = slices.broadcast(source._slices,mode="dim")[0]
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
            nnslices.append(slices.FlatAllSlice(slice, ndim))

        return self._initialize(tuple(nnslices),source._state)

class Flat(repops.UnaryOpRep):
    def _process(self,source,name=None,dim=-1):
        if not source._state & RS_SLICES_KNOWN:
            return
        
        #create new merged dimension
        selpath = dimpaths.identifyUniqueDimPathSource(source,dim)
        if(len(selpath) == 1):
            selpath = dimpaths.extendParentDim(selpath,[s.dims for s in source._slices])

        if(selpath[-1].shape != UNDEFINED and selpath[-2].shape != UNDEFINED):
            shape = selpath[-2].shape * selpath[-1].shape
        else:
            shape = UNDEFINED
        if(name is None):
            name = selpath[-2].name + "_" + selpath[-1].name
        dependent = tuple([left or right for left,right in 
                        izip_longest(selpath[-2].dependent, selpath[-1].dependent[:-1],fillvalue=False)])
        ndim = dimensions.Dim(shape, dependent=dependent, has_missing = selpath[-1].has_missing or selpath[-2].has_missing, name=name)
        
        bcdim = dimensions.Dim(1)

        #find refslices
        refslices = [s for s in source._slices if selpath[-1] in s.dims]

        #process slices
        nslices = []
        for slice in source._slices:
            sdims = slice.dims
            startpos = sdims.matchDimPath(selpath[:-1])
            while(startpos):
                spos = startpos[0]
                flatpos = len(selpath) + spos -1 
                if(len(sdims) <= flatpos or sdims[flatpos] != selpath[-1]):
                    slice = slices.InsertDimSlice(slice,flatpos,bcdim)
                    bcdims = slice.dims[:flatpos] + (selpath[-1],) + slice.dims[flatpos:]
                    plan = [BCCOPY] * len(slice.dims[:flatpos]) + [BCEXIST] + [BCCOPY] * len(slice.dims[flatpos:])
                    slice = slices.BroadcastSlice(slice,[refslices],plan,bcdims)
                slice = slices.FlatDimSlice(slice,flatpos,ndim)
                if(len(startpos) > 1):
                    sdims = slice.dims
                    startpos = sdims.matchDimPath(selpath[::-1])
                else:
                    startpos = []
            nslices.append(slice)
                
        return self._initialize(tuple(nslices),source._state)
       
