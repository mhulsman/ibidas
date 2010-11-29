from collections import defaultdict
from ..utils.multi_visitor import VisitorFactory, NF_ELSE

import manager
import prewalk
_delay_import_(globals(),"..slices")

class RequiredSliceIds(VisitorFactory(prefixes=("require", "func"), 
                                      flags=NF_ELSE), 
                       manager.Pass):
    """Calculates for representor objects in query tree
    the targets (i.e. other representor objects that use their data)"""
    after = set([prewalk.PreOrderWalk])

    @classmethod
    def run(cls, query, pass_results):
        self = cls()
        prewalk = pass_results[prewalk.PreOrderWalk]
        required_ids = defaultdict(set)

        for rep in prewalk:
            #pylint: disable-msg=E1101
            self.require(rep, required_ids[rep], required_ids)

        return required_ids
    
    def requireOpRep(self, rep, req_ids, source_ids):
        rep._req_ids = req_ids
        avail_slice_ids = self.getAvailSliceIds(rep)
        inner_req_ids = self.calcInnerReqIds(rep, req_ids, avail_slice_ids)
        #pylint: disable-msg=E1101
        self.func(rep, inner_req_ids, source_ids)
 
    def requireelse(self, rep, req_ids, source_ids):
        rep.request_ids(req_ids, source_ids)

    def _requirenothing(self, rep, req_ids, source_ids):
        """Source representors have no source slices, so require nothing"""
        rep._req_ids = req_ids
    requirePyRepresentor = _requirenothing
    
    def requirefixate(self, rep, required_ids, source_ids):
        """Fixate always requires all active slices of its source"""
        source = rep._sources[0]
        source_ids[source].update([slice.id for slice in source._active_slices])

    def _funccopy(self, rep, inner_req_ids, source_ids):
        """If the slice performs no real action, required_ids
        can be just copied
        Parameters (also valid for other require functions)
        ----------
        rep : representor object for which required source slices
              should be determined
        required_ids : set of slice ids this representor should deliver
        source_ids : dictionary from source representors to their
                     set of required ids. Should be added to. No check
                     necessary if source is in dict, as it is a defaultdict(set).
        """
        source_ids[rep._sources[0]] |= inner_req_ids

    funcUnaryOpRep = _funccopy

    def funcMultiOpRep(self, rep, inner_req_ids, source_ids):
        """Multi operations have multiple source. Determine which slice ids
        are locally calculated, and which are calculated by the sources.
        If multipe sources have the same id, always takes the most left one.
        """
        inner_req_ids = inner_req_ids.copy() 
        for source in rep._sources:
            all_slices = source._all_slices
            in_ids = set([rid for rid in inner_req_ids if rid in all_slices])
            source_ids[source] |= in_ids        
            inner_req_ids -= in_ids

    def funcJoin(self, rep, inner_req_ids, source_ids):
        self.funcMultiOpRep(rep, inner_req_ids, source_ids) 
        for source, jslices in zip(rep._sources, rep._join_slices):
            if(not (source_ids[source] & jslices)):
                source_ids[source].add(iter(jslices).next())
    
    def funcEquiJoin(self, rep, inner_req_ids, source_ids):
        self.funcMultiOpRep(rep, inner_req_ids, source_ids) 
        source_ids[rep._sources[0]].add(rep._lsliceid)
        source_ids[rep._sources[1]].add(rep._rsliceid)
        for source, jslices in zip(rep._sources, rep._join_slices):
            if(not (source_ids[source] & jslices)):
                source_ids[source].add(iter(jslices).next())
    
    def funcRFilter(self, rep, inner_req_ids, source_ids):
        """Required slices are just copied, as filter
        performs no realias, but only a redim. It does however need
        slices from _constraint_slices to change dimensions."""
        rep.inner_req_ids = inner_req_ids
        source_ids[rep._sources[0]] |= inner_req_ids
        for pos, slice in enumerate(rep._constraint_slices):
            source_ids[rep._sources[pos + 1]].add(slice.id)
   
    def funcifilter(self, rep, inner_req_ids, source_ids):
        """Required slices are just copied, as filter
        performs no realias, but only a redim. It does however need
        slices from _constraint_slices to change dimensions."""
        rep.inner_req_ids = inner_req_ids
        source_ids[rep._sources[0]] |= inner_req_ids
        for pos, slice in enumerate(rep._constraint_slices):
            source_ids[rep._sources[pos + 1]].add(slice.id)

    def funcGroup(self, rep, inner_req_ids, source_ids):
        """Required slices are just copied, as Group
        performs no realias, but only a redim. It does however need
        the _grouper_slice to perform grouping."""
        rep.inner_req_ids = inner_req_ids
        source_ids[rep._sources[0]] |= inner_req_ids
        source_ids[rep._sources[1]].update([slice.id for slice in rep._group_slices])
   
    def funcflat(self, rep, inner_req_ids, source_ids):
        """Required slices are just copied, as Flatten
        performs no realias, but only a redim. It does however need
        one of the flattened slices to perform the flattening."""
        rep.inner_req_ids = inner_req_ids
        if(not inner_req_ids & rep._flat_slices):
            inner_req_ids.add(iter(rep._flat_slices).next())
        source_ids[rep._sources[0]] |= inner_req_ids
              
    def getAvailSliceIds(self, rep):
        avail_slice_ids = set()
        for source in rep._sources:
            avail_slice_ids.update(source._all_slices.keys())
        return avail_slice_ids

    def calcInnerReqIds(self, rep, req_ids, avail_slice_ids):
        inner_req_ids = avail_slice_ids & req_ids
        id_queue = req_ids - inner_req_ids
        
        while(id_queue): #handle remaining slices
            rid = id_queue.pop()
            if(rid in avail_slice_ids):
                inner_req_ids.add(rid) 
                continue
            
            assert rid in rep._all_slices, "ID: " + str(rid) + " required in " \
                                        + str(rep.__class__) + " but not found."
            
            slice = rep._all_slices[rid]
            
            if slice.last_id in avail_slice_ids:
                inner_req_ids.add(slice.last_id)
            elif isinstance(slice, slices.OpSlice):
                id_queue.update(slice.source_ids)
            else:
                raise RuntimeError, "Cannot find source slice for slice " \
                                + str(rid) + " in " + str(rep.__class__)
        
        return inner_req_ids 
