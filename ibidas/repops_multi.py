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
        


        
