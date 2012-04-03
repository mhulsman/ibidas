"""Classes to handle array data with sparse values"""


import numpy
import operator
import array
from itertools import izip_longest

from missing import Missing
import util
_delay_import_(globals(),"cutils")
#_delay_import_(globals(),"ibidas.itypes","Missing")



class FullSparse(numpy.ndarray):
    def __new__(cls,obj):
        res = numpy.array(obj).view(cls)
        return res

    def getVal(self):
        return self.view(numpy.ndarray)
    val = property(fget=getVal)
    
    def _getClasses(self):
        classes = set()
        if(self.dtype == object):
            classes |= set([value.__class__ for value in self.ravel()])
        else:
            classes.add(self.dtype.type)

        return classes
    classes=property(fget=_getClasses)

    def map(self, func, **kwargs):
        out_empty = kwargs.get('out_empty',Missing)
        
        if(kwargs.get('has_missing',True)):
            func = elem_empty_filter(func,out_empty)
        
        otype = kwargs.get('otype',self.dtype)

        if(otype == object):
            res = util.darray([func(elem) for elem in self.ravel()])
            res.shape = self.shape
        elif(self.dtype.char == 'S' or self.dtype.char == 'U' or self.dtype.char == 'V'):
            res = util.darray([func(elem) for elem in self.ravel()],otype)
            res.shape = self.shape
        else:
            func = numpy.frompyfunc(func,1,1)
            res = numpy.cast[otype](func(self))
        
        return FullSparse(res)

    def filter_tomissing(self,filter):
        nval = self.copy()
        nval = numpy.cast[object](nval)
        nval[~filter] = Missing
        return nval.view(FullSparse)

    def hasSparse(self):
        return numpy.equal(self,Missing).any()

    def replace_missing(self,empty_replace=Missing,otype=None):
        if(not empty_replace is Missing):
            missing_filter = numpy.equal(self,Missing)
            res = self.copy()
            res[missing_filter] = empty_replace
        else:
            res = self
        if(not otype is None):
            res = numpy.cast[otype](res)
        return res
    
    def without_missing(self):
        val = self.copy()
        missing_filter = numpy.equal(self, Missing)
        return val[~missing_filter]


    def min(self,has_missing=True,out_empty=Missing):
        if(has_missing):
            val = self.without_missing()
            if len(val) > 0:
                return val.val.min()
            else:
                return out_empty
        else:
            return self.val.min()

    def max(self,has_missing=True,out_empty=Missing):
        if(has_missing):
            val = self.without_missing()
            if len(val) > 0:
                return val.val.max()
            else:
                return out_empty
        else:
            return self.val.max()
    
    def all(self,has_missing=True,out_empty=Missing):
        if(has_missing):
            val = self.without_missing()
            if len(val) > 0:
                return val.val.all()
            else:
                return out_empty
        else:
            return self.val.all()
    
    def any(self,has_missing=True,out_empty=Missing):
        if(has_missing):
            val = self.without_missing()
            if len(val) > 0:
                return val.val.any()
            else:
                return out_empty
        else:
            return self.val.any()




def sparse_concat(*seqs):
    poss = []
    vals = []
    length = 0
    if not seqs:
        return PosSparse([],[],0)
    

    if all([isinstance(seq,numpy.ndarray) for seq in seqs]):
        for seq in seqs:
            if(isinstance(seq,FullSparse)):
                pass
        return numpy.concatenate(seqs) 
    else:
       pass 

    for seq in seqs:
        if not isinstance(seq, PosSparse):
            seq = from_seq(seq)
        assert isinstance(seq,PosSparse), "Cannot convert input to sparse list"
        poss.append(seq.sorted_pos + length)
        length += seq.length
        vals.append(seq.val)

    return PosSparse(numpy.concatenate(vals), numpy.concatenate(poss), shape)

def elem_empty_filter(func,out_empty=Missing):
    def f(elem):
        if elem is Missing:
            return out_empty
        else:
            return func(elem)
    return f

def _inbetween_tile(lpos,apos,nrows,ncols,times):
    steps = (numpy.arange(nrows)+1) * ncols
    templpos = []
    tempapos = []
    steppos = 0
    pospos = 0
    lenlpos = len(lpos)
    while(steppos < nrows and pospos < lenlpos):
        curstepend = steps[steppos]
        beginpospos = pospos
        while(pospos < lenlpos and apos[pospos] < curstepend):
            pospos+=1
        t = apos[beginpospos:pospos]
        tempapos.extend(numpy.tile(t,times) + numpy.repeat(numpy.arange(times) * ncols,len(t)) + steppos * ncols * (times-1))
        t = lpos[beginpospos:pospos]
        templpos.extend(numpy.tile(t,times))
        steppos += 1

    lpos = numpy.cast[int](templpos)
    apos = numpy.cast[int](tempapos)
    return (lpos,apos)


