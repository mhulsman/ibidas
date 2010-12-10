import numpy
import collections
from ..constants import *

_delay_import_(globals(), "..utils","sparse_arrays","cutils","missing")
_delay_import_(globals(), "..utils.missing","Missing")
_delay_import_(globals(), "dimpaths")

def getConvertor(rtype):
    if 'convertor' in rtype.attr:
        return rtype.attr['convertor']
    else:
        return BaseConvertor(set([None.__class__,missing.MissingType]))

class BaseConvertor(object):
    def __init__(self,objectclss):
        self.objectclss = objectclss

    def convert(self,seq,elem_type):
        return seq

class ArrayConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        """Converts sequence to standard format. Converts None
        to Missing values."""
        numpy_type = elem_type.toNumpy()
        return self._convert(elem_type.getFullDimPath(),numpy_type,seq)
   
    def _convert(self,dims,numpy_type,seq):
        cur_numpy_type = numpy_type
        
        depth = dims.contigiousFixedNDims()
        if(not depth):# no fixed dims, allow for one non-fixed dim
            depth = 1
        rest_dims = dims[depth:]
        if(rest_dims):
            cur_numpy_type = object

        if(len(seq.shape) - 1 >= depth):
           print "HOI"
      
        res = []
        for elem in seq:
            if(elem is Missing or elem is None):
                res.append(Missing)
                continue
            #convert to numpy array if needed
            if(not isinstance(elem, numpy.ndarray)):
                if(not isinstance(elem, collections.Sequence)):
                    elem = list(elem)
                elem = cutils.darray(elem,object,depth,1)
            
            #assure correct number of dimensions
            if(len(elem.shape) < depth):
                oshape = elem.shape
                rem_ndims = depth - len(elem.shape) + 1
                elem = cutils.darray(list(elem.ravel()),cur_numpy_type,rem_ndims,rem_ndims)
                elem.shape = oshape + elem.shape[1:]
                assert len(elem.shape) == depth, "Non array values encountered for dims " + str(dims[len(elem.shape):])
            elif(len(elem.shape) > depth):
                oshape = elem.shape
                elem = dimpaths.flatFirstDims(elem,depth)
                elem = cutils.darray([subelem for subelem in elem],object,1,1)
                elem.shape = oshape[:depth]

            if(not isinstance(elem,sparse_arrays.FullSparse)):
                if not elem.dtype == cur_numpy_type:
                    elem = numpy.cast[numpy_type](elem)
                elem = sparse_arrays.FullSparse(elem)

            if(rest_dims):
                elem = self._convert(rest_dims,numpy_type,elem)
            res.append(elem)

        nseq = cutils.darray(res,object,depth+1,1)
        seq = sparse_arrays.FullSparse(nseq)
        return seq

class StringConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        #deduplicate
        x = seq[:100]
        if(len(x) * 0.6 > len(set(x))):
            r = dict()
            res = []
            for elem in seq:
                if(elem is Missing or elem is None):
                    res.append(Missing)
                else:
                    try:
                        elem = r[elem]
                    except KeyError:
                        r[elem] = elem
                    res.append(elem)
            nseq = cutils.darray(res)
            seq = sparse_arrays.FullSparse(nseq)
        return seq


class StringFloatConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
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
        seq = sparse_arrays.FullSparse(nseq)
        return seq

class StringIntegerConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
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
        seq = sparse_arrays.FullSparse(nseq)
        return seq
 
class SetConvertor(BaseConvertor):

    def convert(self,seq,elem_type):
        if(set in self.objectclss): 
            if(elem_type.has_missing):
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
        seq = sparse_arrays.FullSparse(nseq)
        return seq



class DictConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        for fieldname, subtype in zip(rtype.fieldnames, rtype.subtypes):
            def getname(elem):
                try:
                    return elem[fieldname]
                except (KeyError, TypeError):
                    return Missing
            
            subseq = cutils.darray([getname(elem) for elem in seq],subtype.toNumpy())
            columns.append(self.execFreeze(subtype, subseq))
        nseq = cutils.darray(zip(*columns))
        seq = sparse_arrays.FullSparse(nseq)
        return seq

class NamedTupleConvertor(BaseConvertor):
    def convert(self,seq,elem_type):
        for fieldname, subtype in zip(rtype.fieldnames, rtype.subtypes):
            def getname(elem):
                try:
                    return getattr(elem,fieldname)
                except (KeyError, TypeError):
                    return Missing
            subseq = cutils.darray([getname(elem) for elem in seq],subtype.toNumpy())
            columns.append(self.execFreeze(subtype, subseq))
        nseq = cutils.darray(zip(*columns))
        seq = sparse_arrays.FullSparse(nseq)
        return seq

