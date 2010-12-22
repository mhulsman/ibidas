import operator

from constants import *
import repops

_delay_import_(globals(),"utils","util")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"representor")
_delay_import_(globals(),"wrappers","wrapper_py")

class Broadcast(repops.MultiOpRep):
    def process(self,sources, mode="dim", partial=False):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        nslices = []
        for bcslices in util.zip_broadcast(*[source._slices for source in sources]):
            nslices.extend(slices.broadcast(bcslices,mode,partial))
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
            lslice,rslice = slices.broadcast(binslices,mode,partial=True)
            nslices.append(slices.BinElemOpSlice(lslice,rslice,op,pos))
        return self.initialize(tuple(nslices),state)
        


class Filter(repops.MultiOpRep):
    def __init__(self,source,constraint,dim=None):
        if(not isinstnace(constraint,representor.Representor)):
            constraint = wrapper_py.rep(lsource)
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
            assert dim is None, "Cannot use bool or missing data type with specified filter dimension. Constraint dimension already specifies dimension."
            ndim = dimensions.Dim(UNDEFINED, 
                                  len(cslice.dims) - 1, False,
                                  name = cslice.dims[-1].name
                                  )

            nslices = [slices.BoolFilter(slice,cslice,ndim) for slice in source._slices]
        else:
            pass
                         
        return self.initialize(tuple(nslices),source._state)
                    





