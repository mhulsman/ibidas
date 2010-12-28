import operator

from constants import *
import repops

_delay_import_(globals(),"utils","util")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"representor")
_delay_import_(globals(),"wrappers","wrapper_py")
_delay_import_(globals(),"itypes","rtypes","dimensions","dimpaths")

class Broadcast(repops.MultiOpRep):
    def process(self,sources, mode="dim"):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        nslices = []
        for bcslices in util.zip_broadcast(*[source._slices for source in sources]):
            nslices.extend(slices.broadcast(bcslices,mode))
        return self.initialize(tuple(nslices),state)


class Combine(repops.MultiOpRep):
    def process(self,sources):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        nslices = sum([source._slices for source in sources],tuple())
        return self.initialize(nslices,state)

class Binop(repops.MultiOpRep):
    def __init__(self, lsource, rsource, op):
        if(not isinstance(lsource, representor.Representor)):
            lsource = repops.PlusPrefix(wrapper_py.rep(lsource))
        if(not isinstance(rsource, representor.Representor)):
            rsource = repops.PlusPrefix(wrapper_py.rep(rsource))
        repops.MultiOpRep.__init__(self,(lsource,rsource),op=op)

    def process(self, sources, op):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_TYPES_KNOWN:
            return

        if(any([isinstance(source,repops.PlusPrefix) for source in sources])):
            mode = "pos"
        else:
            mode = "dim"

        source_slices = [source._slices for source in sources]
        nslices = []
        for pos, binslices in enumerate(util.zip_broadcast(*source_slices)):
            lslice,rslice = slices.broadcast(binslices,mode)
            nslices.append(slices.BinElemOpSlice(lslice,rslice,op,pos))
        return self.initialize(tuple(nslices),state)
        


class Filter(repops.MultiOpRep):
    def __init__(self,source,constraint,dim=0):
        if(not isinstance(constraint,representor.Representor)):
            constraint = repops.PlusPrefix(wrapper_py.rep(constraint))
        repops.MultiOpRep.__init__(self,(source,constraint),dim=dim)

    def process(self,sources,dim):
        source,constraint = sources
        if not source._state & RS_SLICES_KNOWN:
            return
        if not constraint._state & RS_TYPES_KNOWN:
            return

        assert len(constraint._slices) == 1, "Filter constraint should have 1 slice"
        cslice = constraint._slices[0]

        if(isinstance(cslice.type,rtypes.TypeBool)):
            assert dim == 0, "Cannot use bool or missing data type with specified filter dimension. Constraint dimension already specifies dimension."
            assert cslice.dims, "Constraint should have at least one dimension"
            ndim = dimensions.Dim(UNDEFINED,tuple(), False, name = cslice.dims[-1].name)
            seldim = cslice.dims[-1]
            cslice = slices.PackArraySlice(cslice,1)
        else:
            dim_suffix = dimpaths.identifyDimPath([s.dims for s in source._slices], dim)
            seldim = dim_suffix[-1]
            
            if(isinstance(cslice.type, (rtypes.TypeArray, rtypes.TypeSlice))):
                if(isinstance(cslice.type, rtypes.TypeArray)):
                    assert len(cslice.type.dims) == 1, "Filter array should be 1-dimensional"
                    assert isinstance(cslice.type.subtypes[0], rtypes.TypeInteger) and \
                                not isinstance(cslice.type.subtypes[0], rtypes.TypeBool), \
                                "Subtype of an array should be integer (excluding bool)"
                    ndim = cslice.type.dims[0]
                else:
                    ndep = max(len(seldim.dependent),len(cslice.dims))
                    shape = UNDEFINED
                    ndim = dimensions.Dim(UNDEFINED, tuple(), False,name = "f" + cslice.name)
            elif(isinstance(cslice.type, rtypes.TypeInteger)):
                 ndim = None
            else:
                raise RuntimeError, "Unknown constraint type in filter: " + str(cslice.type)

        if(isinstance(constraint, repops.PlusPrefix)):
            mode = "pos"
        else:
            mode = "dim"

        nslices = []
        for slice in source._slices:
            while seldim in slice.dims:
                slice = slices.filter(slice, cslice, seldim, ndim, mode)
            nslices.append(slice)
        return self.initialize(tuple(nslices),source._state)
                    





