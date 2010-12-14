from constants import *
import repops

_delay_import_(globals(),"itypes","rtypes","dimpaths","dimensions")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"utils","util")

class UnpackArray(repops.UnaryOpRep):
    def process(self,source, name = None, ndim=None):
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

        return self.initialize(tuple(nslices),RS_CHECK)

class DimRename(repops.UnaryOpRep):#{{{
    def process(self, source, *names, **kwds):
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
        return self.initialize(tuple(nslices),source._state)#}}}

class InsertDim(repops.UnaryOpRep):
    def process(self, source, insertpoint, name=None):
        if not source._state & RS_SLICES_KNOWN:
            return
        
        assert len(dimpaths.uniqueDimPath([s.dims for s in source._slices],only_complete=False)) >= insertpoint, "No unique dimpath covering dim insertion point"
        ndim = dimensions.Dim(1,variable=0,has_missing=False,name=None)
        nslices = []
        for slice in source._slices:
            slice = slices.InsertDimSlice(slice,insertpoint,ndim)
            nslices.append(slice)
        return self.initialize(tuple(nslices),source._state)

class Broadcast(repops.MultiOpRep):
    def process(self,sources, mode="full", partial=False):
        if not source._state & RS_SLICES_KNOWN:
            return
        lengths = set([len(source._slices) for source in sources])
        lengths.pop(1)
        assert len(lengths) == 1, "Number of slices in sources to broadcast should be equal (or 1)"
        length = lengths.pop()

        all_slices = []
        for source in sources:
            if(len(source._slices) == 1):
                all_slices.append(source._slices * length)
            else:
                all_slices.append(source._slices)

        nslices = sum([slices.broadcast(bcslices,mode,partial) for bcslices in zip(*all_slices)],[])

        state = reduce(operator.__and__,[source._state for source in sources])
        return self.initialize(tuple(nslices),state)

@repops.delayable()
def rarray(source, dim=None, ndim=1):
    return repops.ApplyFuncRep(source, repops.apply_slice, slices.PackArraySlice, dim, ndim=1)

@repops.delayable()
def rlist(source, dim=None):
    return repops.ApplyFuncRep(source, repops.apply_slice, slices.PackListSlice, dim)

