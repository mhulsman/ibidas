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
        return (rtype.data_state != DATA_FROZEN)
    needFreezeTypeUnknown=need_freeze
    needFreezeTypeDict=need_freeze
    needFreezeTypeArray=need_freeze
    needFreezeTypeSlice=need_freeze

    def noneed_freeze(self, rtype):
        return False
    needFreezeTypeString=noneed_freeze
    needFreezeTypeScalar=noneed_freeze
    needFreezeTypeSet=noneed_freeze
    
    def needFreezeTypeTuple(self,rtype):
        return any([self.needFreeze(subtype)for subtype in rtype.subtypes])

    
    #freeze a type and its subtypes (adapt its representation, not the data!)
    def freezeTypeUnknown(self, rtype):
        if self.needFreeze(rtype):
            return rtype.copy(data_state=DATA_FROZEN)
        else:
            return rtype

    def freezeTypeTuple(self, rtype):
        if(self.needFreeze(rtype)):
            res = rtype.copy()
            res.subtypes = [self.freezeType(subtype) for subtype in rtype.subtypes]
        else:
            res = rtype
        return res
    
    def freezeTypeDict(self, rtype):
        if(self.needFreeze(rtype)):
            res = rtype.copy(data_state=DATA_FROZEN)
            res.subtypes = [self.freezeType(subtype) for subtype in rtype.subtypes]
        else:
            res = rtype
        return res

    def freezeTypeArray(self,rtype):
        if(self.needFreeze(rtype)):
            res = rtype.copy(data_state=DATA_FROZEN)
            res.subtypes = [self.freezeType(subtype) for subtype in rtype.subtypes]
        else:
            res = rtype
        return res
    
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
        return cutils.darray([self.ftype(elem) 
                for elem in obj]).view(util.farray) 
    ftypelist = _ftypesequence

    def ftypendarray(self, obj):
        if(obj.dtype == object):
            return cutils.darray([self.ftype(elem) 
                for elem in obj]).view(util.farray) 
        else:
            return obj.view(util.farray)

    def ftypeset(self, obj):
        return frozenset(obj)

    def ftypeslice(self, obj):
        return slicetuple(obj)

    def map(self, func, seq, parent_type):
        if(parent_type.has_missing):
            seqres = []
            for elem in seq:
                if(elem is Missing):
                    seqres.append(Missing)
                else: 
                    seqres.append(func(elem))
        else:
            seqres = [func(elem) for elem in seq]
        return seqres 
    

    def execFreezeTypeUnknown(self, rtype, seq):
        if(self.needFreeze(rtype)):
            seq = seq.map(self.ftype,has_missing=rtype.has_missing)
        return seq
   
    def execFreezeTypeScalar(self,rtype,seq):
        return seq


    def execFreezeTypeSlice(self,rtype,seq):
        if(self.needFreeze(rtype)):
            seq = seq.map(self.ftypeslice,has_missing=rtype.has_missing)
        return seq
        

    def execFreezeTypeTuple(self,rtype,seq):
        if(self.needFreeze(rtype)):
            if(not rtype.subtypes):
                return self.execFreezeTypeUnknown(rtype,seq)
            
            columns = []
            if(len(rtype.subtypes) > rtype.min_len):
                l = seq.map(len, otype=int, out_empty = 0, has_missing=self.detector.hasMissing())

            for pos, subtype in enumerate(rtype.subtypes):
                f = operator.itemgetter(pos)
                if(pos < rtype.min_len):
                    subseq = seq.map(f,otype=object,out_empty=Missing,has_missing=rtype.has_missing)
                else:
                    subseq = seq.sparse_filter(l > pos).map(f,out_empty=Missing,otype=object,has_missing=True)
                columns.append(self.execFreeze(subtype, subseq))

            nseq = cutils.darray(zip(*columns))
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

    #        nseq = cutils.darray(zip(*columns))
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

    #        nseq = cutils.darray(zip(*columns))
    #        nseq.shape = seq.shape
    #        seq = sparse_arrays.FullSparse(nseq)
    #    return seq

    def execFreezeTypeArray(self,rtype,seq):
        if(self.needFreeze(rtype)):
            subtype = rtype.subtypes[0]
            if(subtype.needFreeze(rtype)):
                def subfreeze(elem):
                    elem = self.execFreeze(subtype,elem)
                    elem = elem.view(util.farray)
                    return elem
            else:
                def subfreeze(elem):
                    ele = elem.view(util.farray)
                    return elem
            seq = seq.map(subfreeze,out_empty=Missing,otype=object,has_missing=rtype.has_missing)
        return seq
     
    def execFreezeTypeString(self,rtype,seq):
        return seq
    
    def execFreezeTypeSet(self, rtype, seq):
        return seq
    
    
    
    #Determine if necessary to freeze a type (or its subtypes)    
    def need_unfreeze(self, rtype):
        return (rtype.data_state == DATA_FROZEN)
    needUnfreezeTypeUnknown=need_unfreeze
    needUnfreezeTypeDict=need_unfreeze
    needUnfreezeTypeSlice=need_unfreeze

    def noneed_unfreeze(self, rtype):
        return False
    needUnreezeTypeString=noneed_unfreeze
    needUnfreezeTypeScalar=noneed_unfreeze
    needUnfreezeTypeSet=noneed_unfreeze
    needUnfreezeTypeTuple=noneed_unfreeze
   
    #freeze a type and its subtypes (adapt its representation, not the data!)
    def unfreezeTypeUnknown(self, rtype):
        if self.needUnfreeze(rtype):
            return rtype.copy(data_state=DATA_NORMAL)
        else:
            return rtype
    
    #freeze standard python objects
    def uftypeobject(self, obj):
        if(hasattr(obj, "__unfreeze__")):
            return obj.__unfreeze__()
        else:
            return obj

    def _uftypenofreeze(self,obj):
        return obj
    uftypeint=_uftypenofreeze
    uftypefloat=_uftypenofreeze
    uftypestr=_uftypenofreeze
    uftypeunicode=_uftypenofreeze
    uftypetuple=_uftypenofreeze

    def uftypefarray(self, obj):
        return sparse_arrays.FullSparse(obj)

    def uftypeslicetuple(self, obj):
        return obj.getSlice()

    def execUnfreezeTypeUnknown(self, rtype, seq):
        if(self.needUnfreeze(rtype)):
            seq = seq.map(self.uftype,has_missing=rtype.has_missing)
        return seq
   
    def execUnfreezeTypeScalar(self,rtype,seq):
        return seq


    def execUnfreezeTypeSlice(self,rtype,seq):
        if(self.needFreeze(rtype)):
            seq = seq.map(self.uftypeslice,has_missing=rtype.has_missing)
        return seq
        
    def execUnfreezeTypeTuple(self,rtype,seq):
        return seq


    def execUnfreezeTypeArray(self,rtype,seq):
        
        if(self.needUnfreeze(rtype)):
            def subunfreeze(elem):
                seq = sparse_arrays.FullSparse(elem)
                return seq
            seq = seq.map(subunfreeze,out_empty=Missing,otype=object,has_missing=rtype.has_missing)
        return seq
     
    def execUnfreezeTypeString(self,rtype,seq):
        return seq
    
    def execUnfreezeTypeSet(self, rtype, seq):
        return seq
    
   
freeze_protocol = RTypeFreezeProtocol()

