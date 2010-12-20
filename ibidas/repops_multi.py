import operator

from constants import *
import repops

_delay_import_(globals(),"slices")
class Broadcast(repops.MultiOpRep):
    def process(self,sources, mode="dim", partial=False):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return
        lengths = set([len(source._slices) for source in sources])
        lengths.discard(1)
        assert len(lengths) <= 1, "Number of slices in sources to broadcast should be equal (or 1)"
        if(lengths):
            length = lengths.pop()
        else:
            length = 1

        all_slices = []
        for source in sources:
            if(len(source._slices) == 1):
                all_slices.append(source._slices * length)
            else:
                all_slices.append(source._slices)

        nslices = sum([slices.broadcast(bcslices,mode,partial) for bcslices in zip(*all_slices)],[])

        return self.initialize(tuple(nslices),state)


class Combine(repops.MultiOpRep):
    def process(self,*sources, **kwds):
        state = reduce(operator.__and__,[source._state for source in sources])
        if not state & RS_SLICES_KNOWN:
            return

        nslices = sum([source._slices for source in sources],tuple())
        return self.initialize(nslices,state)

