import operator

from ..constants import *
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, NF_ELSE
from ..utils.missing import Missing

_delay_import_(globals(),"..utils","cutils","util","sparse_arrays")

class slicetuple(tuple):
    def __new__(self,s):
        res = tuple.__new__(self,(s.start,s.stop,s.step))
        return res

    def getslice(self):
        return slice(*self)


class RTypeFreezeProtocol(VisitorFactory(prefixes=("needFreeze", "freeze","execFreeze","ftype","needUnfreeze","unfreeze","execUnfreeze","uftype"), 
                                                        flags=NF_ELSE)):

    #Determine if necessary to freeze a type (or its subtypes)    
    def need_freeze(self, rtype):
        return True
    needFreezeTypeAny=need_freeze
    needFreezeTypeDict=need_freeze
    needFreezeTypeArray=need_freeze
    needFreezeTypeSlice=need_freeze
    needFreezeTypeSet=need_freeze

    def noneed_freeze(self, rtype):
        return False
    needFreezeTypeString=noneed_freeze
    needFreezeTypeScalar=noneed_freeze
    
    def needFreezeTypeTuple(self,rtype):
        return any([self.needFreeze(subtype)for subtype in rtype.subtypes])
    
    #freeze standard python objects
    def ftypeobject(self, obj):
        if(hasattr(obj, "__freeze__")):
            return obj.__freeze__()
        else:
            return obj

    def _ftypenofreeze(self,obj):
        return obj
    ftypeint=_ftypenofreeze
    ftypefloat=_ftypenofreeze
    ftypestr=_ftypenofreeze
    ftypeunicode=_ftypenofreeze

    def ftypetuple(self, obj):
        return tuple([self.ftype(elem) for elem in obj])

    def _ftypesequence(self, obj):
        return util.darray([self.ftype(elem) 
                for elem in obj]).view(util.farray) 
    ftypelist = _ftypesequence

    def ftypendarray(self, obj):
        if(obj.dtype == object):
            return util.darray([self.ftype(elem) 
                for elem in obj]).view(util.farray) 
        else:
            return obj.view(util.farray)

    def ftypeset(self, obj):
        return frozenset(obj)

    def ftypeslice(self, obj):
        return slicetuple(obj)

    def map(self, func, parent_type, seq):
        if(parent_type.has_missing):
            seqres = []
            for elem in seq:
                if(elem is Missing):
                    seqres.append(Missing)
                else: 
                    seqres.append(func(elem))
        else:
            seqres = [func(elem) for elem in seq]
        
        seqres = util.darray(seqres,seq.dtype)
        return seqres 
    

    def execFreezeTypeUnknown(self, ptype, seq):
        seq = self.map(self.ftype, ptype, seq)
        return seq
   
    def execFreezeTypeScalar(self, ptype, seq):
        return seq


    def execFreezeTypeSlice(self, ptype, seq):
        return self.map(self.ftypeslice, ptype, seq)
        

    def execFreezeTypeTuple(self, ptype, seq):
        if(not ptype.subtypes):
            return self.execFreezeTypeUnknown(ptype, seq)
        else:
            return self.map(self.ftypetuple, ptype, seq)
        
        columns = []
        if(len(ptype.subtypes) > ptype.min_len):
            l = self.map(len, otype=int, out_empty = 0, has_missing=self.detector.hasMissing())

        for pos, subtype in enumerate(rtype.subtypes):
            f = operator.itemgetter(pos)
            if(pos < rtype.min_len):
                subseq = seq.map(f,otype=object,out_empty=Missing,has_missing=rtype.has_missing)
            else:
                subseq = seq.sparse_filter(l > pos).map(f,out_empty=Missing,otype=object,has_missing=True)
            columns.append(self.execFreeze(subtype, subseq))

        nseq = util.darray(zip(*columns))
        nseq.shape = seq.shape
        seq = sparse_arrays.FullSparse(nseq)
        return seq
    
    #def execFreezeTypeDict(self,rtype,seq):
    #    if(self.needFreeze(rtype)):
    #        if(not rtype.subtypes):
    #            return self.execFreezeTypeUnknown(rtype,seq)
    #        
    #        columns = []
    #        if(len(rtype.subtypes) > rtype.min_len):
    #            l = seq.map(len, otype=int, out_empty = 0, has_missing=self.detector.hasMissing())

    #        for pos, (fieldname, subtype) in enumerate(zip(rtype.fieldnames, rtype.subtypes)):
    #            def getname(elem):
    #                try:
    #                    return elem[name]
    #                except (KeyError, TypeError):
    #                    return Missing
    #            subseq = seq.map(getname,otype=object,out_empty=Missing,has_missing=rtype.has_missing)
    #            columns.append(self.execFreeze(subtype, subseq))

    #        nseq = util.darray(zip(*columns))
    #        nseq.shape = seq.shape
    #        seq = sparse_arrays.FullSparse(nseq)
    #    return seq
    #
    #def execFreezeTypeNamedTuple(self,rtype,seq):
    #    if(self.needFreeze(rtype)):
    #        if(not rtype.subtypes):
    #            return self.execFreezeTypeUnknown(rtype,seq)
    #        
    #        columns = []
    #        if(len(rtype.subtypes) > rtype.min_len):
    #            l = seq.map(len, otype=int, out_empty = 0, has_missing=self.detector.hasMissing())

    #        for pos, (fieldname, subtype) in enumerate(zip(rtype.fieldnames, rtype.subtypes)):
    #            def getname(elem):
    #                try:
    #                    return elem.name
    #                except (KeyError, TypeError):
    #                    return Missing
    #            subseq = seq.map(getname,otype=object,out_empty=Missing,has_missing=rtype.has_missing)
    #            columns.append(self.execFreeze(subtype, subseq))

    #        nseq = util.darray(zip(*columns))
    #        nseq.shape = seq.shape
    #        seq = sparse_arrays.FullSparse(nseq)
    #    return seq

    def execFreezeTypeArray(self, ptype, seq):
        subtype = ptype.subtypes[0]
        if(self.needFreeze(subtype)):
            def subfreeze(elem):
                elem = self.execFreeze(subtype, elem)
                elem = elem.view(util.farray)
                return elem
        else:
            def subfreeze(elem):
                elem = elem.view(util.farray)
                return elem
        seq = self.map(subfreeze, ptype, seq)
        return seq
     
    def execFreezeTypeString(self,ptype,seq):
        return seq
    
    def execFreezeTypeSet(self, ptype, seq):
        seq = self.map(frozenset, ptype, seq)
        return seq
    
freeze_protocol = RTypeFreezeProtocol()

