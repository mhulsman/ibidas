import sparse_arrays

class BaseConvertor(object):
    def __init__(self,objectclss):
        self.objectclss = objectclss

    def convert(self,elem_type,seq):
        """Converts sequence to standard format. Converts None
        to Missing values."""
        numpy_type = elem_type.toNumpy()
        
        if(not isinstance(seq,(sparse_arrays.FullSparse,sparse_arrays.PosSparse))):
            if(not isinstance(seq, collections.Sequence)):
                seq = list(seq)
            if(not (isinstance(seq,numpy.ndarray) and seq.dtype == numpy_type)):
                seq = cutils.darray(seq,numpy_type)
            seq = sparse_arrays.FullSparse(seq)
            
        if(elem_type.has_missing):
            if(None.__class__ in self.objectclss):
                seq = seq.full()
                seq[numpy.equal(seq, None)] = Missing
        return seq


class StringConvertor(BaseConvertor):
    def convert(self,elem_type,seq):
        seq = BaseConvertor.convert(self,elem_type,seq)
        
        #deduplicate
        x = seq.ravel()[:100]
        if(len(x) * 0.6 > len(set(x))):
            r = dict()
            res = []
            for elem in seq.ravel():
                try:
                    elem = r[elem]
                except KeyError:
                    r[elem] = elem
                res.append(elem)
            nseq = cutils.darray(res)
            nseq.shape = seq.shape
            seq = sparse_arrays.FullSparse(nseq)
        return seq


class StringFloatConvertor(BaseConvertor):
    def convert(self,elem_type,seq):
        seq = BaseConvertor.convert(self,elem_type,seq)
        
        nan = float("nan")
        if(elem_type.has_missing):
            seqres = []
            for elem in seq:
                if(not elem):
                    seqres.append(nan)
                else: 
                    seqres.append(float(elem))
        else:
            seqres = [float(elem) for elem in seq]
         
        nseq = cutils.darray(seqres,elem_type.toNumpy())
        nseq.shape = seq.shape
        seq = sparse_arrays.FullSparse(nseq)
        return seq

class StringIntegerConvertor(BaseConvertor):
    def convert(self,elem_type,seq):
        seq = BaseConvertor.convert(self,elem_type,seq)
        
        if(elem_type.has_missing):
            seqres = []
            for elem in seq:
                if(not elem):
                    seqres.append(Missing)
                else: 
                    seqres.append(int(elem))
        else:
            seqres = [int(elem) for elem in seq]
         
        nseq = cutils.darray(seqres,elem_type.toNumpy())
        nseq.shape = seq.shape
        seq = sparse_arrays.FullSparse(nseq)
        return seq
 
class SetConvertor(BaseConvertor):
    def convert(self,elem_type,seq):
        seq = BaseConvertor.convert(self,elem_type,seq)
        if(set in self.objectclss): 
            if(parent_type.has_missing):
                seqres = []
                for elem in seq:
                    if(elem is None or elem is Missing):
                        seqres.append(Missing)
                    else: 
                        seqres.append(frozenset(elem))
            else:
                seqres = [frozenset(elem) for elem in seq]
        else:
            seqres = seq
        nseq = cutils.darray(seqres,elem_type.toNumpy())
        nseq.shape = seq.shape
        seq = sparse_arrays.FullSparse(nseq)
        return seq



class DictConvertor(BaseConvertor):
    def convert(self,elem_type,seq):
        seq = BaseConvertor.convert(self,elem_type,seq)
        for fieldname, subtype in zip(rtype.fieldnames, rtype.subtypes):
            def getname(elem):
                try:
                    return elem[fieldname]
                except (KeyError, TypeError):
                    return Missing
            
            subseq = seq.map(getname,otype=subtype.toNumpy(),out_empty=Missing,has_missing=subtype.has_missing)
            columns.append(self.execFreeze(subtype, subseq))
        nseq = cutils.darray(zip(*columns))
        nseq.shape = seq.shape
        seq = sparse_arrays.FullSparse(nseq)
        return seq

class NamedTupleConvertor(BaseConvertor):
    def convert(self,elem_type,seq):
        seq = BaseConvertor.convert(self,elem_type,seq)
        for fieldname, subtype in zip(rtype.fieldnames, rtype.subtypes):
            def getname(elem):
                try:
                    return getattr(elem,fieldname)
                except (KeyError, TypeError):
                    return Missing
            
            subseq = seq.map(getname,otype=subtype.toNumpy(),out_empty=Missing,has_missing=subtype.has_missing)
            columns.append(self.execFreeze(subtype, subseq))
        nseq = cutils.darray(zip(*columns))
        nseq.shape = seq.shape
        seq = sparse_arrays.FullSparse(nseq)
        return seq

