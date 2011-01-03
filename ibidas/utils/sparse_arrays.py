"""Classes to handle array data with sparse values"""


import numpy
import operator
import array
from itertools import izip_longest

from missing import Missing

_delay_import_(globals(),"cutils")
#_delay_import_(globals(),"ibidas.itypes","Missing")

class PosSparse(object):
    def __init__(self, val,sorted_pos=None, shape=None):
        val = cutils.darray(val)
        self.val = val.ravel()

        if(not shape is None):
            assert numpy.multiply.reduce(shape) >= len(self.val),\
                "Given shape too small for given data"

        if(sorted_pos is None):
            self.sorted_pos = numpy.arange(len(val),dtype=int)
            if(shape is None):
                shape = val.shape
        else:
            self.sorted_pos = numpy.cast[int](sorted_pos)
            assert len(self.sorted_pos) == len(self.val),\
                      "Sorted positions too short for given data"
            if(shape is None):
                shape = (self.sorted_pos.max() + 1,)

        self._shape = shape

    def __len__(self):
        if(self._shape):
            return self._shape[0]
        else:
            return 0

    def _getShape(self):
        return self._shape
    shape=property(fget=_getShape)
    
    def _getDType(self):
        if self.hasSparse():
            return numpy.dtype("object")
        else:
            return self.val.dtype
    dtype=property(fget=_getDType)

    def _getClasses(self):
        classes = set()
        if(self.val.dtype == object):
            classes |= set([value.__class__ for value in self.val])
        else:
            classes.add(self.val.dtype.type)

        if self.hasSparse():
            classes.add(Missing.__class__)
        return classes
    classes=property(fget=_getClasses)

    def __contains__(self, val):
        return (val is Missing and self.hasSparse()) \
               or val in self.val

    def __add__(self, other):
        return self._binop(other, "__add__")

    def __radd__(self, other):
        return self._binop(other, "__radd__")

    def __sub__(self, other):
        return self._binop(other, "__sub__")

    def __rsub__(self, other):
        return self._binop(other, "__rsub__")

    def __mul__(self,other):
        return self._binop(other, "__mul__")

    def __rmul__(self,other):
        return self._binop(other, "__rmul__")

    def __mod__(self, other):
        return self._binop(other, "__mod__")

    def __rmod__(self,other):
        return self._binop(other, "__rmod__")

    def __div__(self,other):
        return self._binop(other, "__div__")

    def __rdiv__(self,other):
        return self._binop(other, "__rdiv__")

    def __floordiv__(self,other):
        return self._binop(other, "__floordiv__")
    
    def __rfloordiv__(sef,other):
        return self._binop(other, "__rfloordiv__")

    def __and__(self,other):
        return self._binop(other, "__and__")

    def __rand__(self,other):
        return self._binop(other, "__rand__")

    def __or__(self,other):
        return self._binop(other, "__or__")

    def __ror__(self,other):
        return self._binop(other, "__ror__")

    def __xor__(self,other):
        return self._binop(other, "__xor__")

    def __rxor__(self,other):
        return self._binop(other, "__rxor__")

    def __lt__(self,other):
        return self._binop(other, "__lt__")

    def __gt__(self,other):
        return self._binop(other, "__gt__")

    def __le__(self,other):
        return self._binop(other, "__le__")

    def __ge__(self,other):
        return self._binop(other, "__ge__")

    def __ne__(self,other):
        return self._binop(other, "__ne__")

    def __eq__(self,other):
        return self._binop(other, "__eq__")

    def __pow__(self,other):
        return self._binop(other, "__pow__")

    def __rpow__(self,other):
        return self._binop(other, "__rpow__")

    def __lshift__(self,toher):
        return self._binop(other, "__lshift__")

    def __rlshift__(self,other):
        return self._binop(other, "__rlshift__")

    def __rshift__(self,other):
        return self._binop(other, "__rshift__")

    def __rrshift__(self,other):
        return self._binop(other, "__rrshift__")

    def __divmod__(self,other):
        return self._binop(other, "__divmod__")

    def __rdivmod__(self,other):
        return self._binop(other, "__rdivmod__")

    def __pos__(self):
        return PosSparse(self.val.__pos__(), self.sorted_pos, self._shape)
    
    def __neg__(self):
        return PosSparse(self.val.__neg__(), self.sorted_pos, self._shape)

    def __invert__(self):
        return PosSparse(self.val.__invert__(), self.sorted_pos, self._shape)

    def __abs__(self):
        return PosSparse(self.val.__abs__(), self.sorted_pos, self._shape)

    def _binop(self, other, func):
        if not isinstance(other, (PosSparse,FullSparse)) and operator.isSequenceType(other):
           other = FullSparse(other)
        
        nempty = getattr(Missing,func)(Missing)
        if not isinstance(other, (PosSparse,FullSparse)):
            nval = getattr(self.val,func)(other)
            nsorted_pos = self.sorted_pos
            nshape = self.shape
            ndtype=nval.dtype 
        else:
            lpos1 = numpy.arange(len(self.sorted_pos))
            apos1 = self.sorted_pos
            lpos2 = numpy.arange(len(other.sorted_pos))
            apos2 = other.sorted_pos
            if self.shape != other.shape:
                etop1 = numpy.multiply.reduce(self.shape)
                etop2 = numpy.multiply.reduce(other.shape)
                ebottom = 1
                nshape = []
                for sdim,odim in izip_longest(self.shape[::-1],other.shape[::-1],fillvalue=1):
                    if(sdim == odim):
                        pass
                    elif(sdim == 1):
                        if(etop1 == 1):
                            apos1 = numpy.tile(apos1,odim) + numpy.repeat(numpy.arange(odim) * ebottom,len(lpos1))
                            lpos1 = numpy.tile(lpos1,odim)
                        else:
                            lpos1, apos1 = _inbetween_tile(lpos1,apos1,etop1,ebottom,odim)
                    elif(odim == 1):
                        if(etop2 == 1):
                            apos2 = numpy.tile(apos2,sdim) + numpy.repeat(numpy.arange(sdim) * ebottom,len(lpos2))
                            lpos2 = numpy.tile(lpos2,sdim)
                        else:
                            lpos2,apos2 = _inbetween_tile(lpos2,apos2,etop2,ebottom,sdim)
                    else:
                        raise ValueError, "Shape mismatch: objects cannot be broadcasted to single shape"
                    ebottom *= max(sdim,odim)
                    etop1 /= sdim
                    etop2 /= odim
                    nshape.insert(0,max(sdim,odim))
            else:
                nshape = self.shape
            aposu = numpy.union1d(apos1,apos2)
            aposu_pos = dict(zip(aposu,numpy.arange(len(aposu))))
            apos2_lpos2 = dict(zip(apos2,lpos2))
            
            #indexes in both
            common_idx = []
            common_lpos1 = []
            common_lpos2 = []
            
            #indexes only in self
            self_idx = []
            self_lpos1 = []
            
            #indexes only in other
            other_idx = []
            other_lpos2 = []

            for ap1,lp1 in zip(apos1,lpos1):
                if(ap1 in apos2_lpos2):
                    common_idx.append(aposu_pos[ap1])
                    common_lpos1.append(lp1)
                    lp2 = apos2_lpos2[ap1]
                    common_lpos2.append(lp2)
                else:
                    self_idx.append(aposu_pos[ap1])
                    self_lpos1.append(lp1)
            
            common_set = set(common_idx)
            for ap2,lp2 in zip(apos2,lpos2):
                if(not ap2 in common_set):    
                    other_idx.append(aposu_pos[ap2])
                    other_lpos2.append(lp2)
            
            nval = numpy.zeros((len(aposu),),dtype=object)
            tmp = getattr(self.val[common_lpos1],func)(other.val[common_lpos2])
            nval[common_idx] = tmp
            nval[self_idx] = getattr(self.val[self_lpos1],func)(Missing)
            nval[other_idx] = getattr(Missing,func)(other.val[other_lpos2])
            ndtype = tmp.dtype
            nsorted_pos = aposu 
        
        res = PosSparse(nval, nsorted_pos, nshape)
        if(not nempty is Missing):
            res = res.full(empty_replace=nempty)
        else:
            res = res.clean(dtype=ndtype)
        return res

    def hasSparse(self):
        return len(self.sorted_pos) < numpy.multiply.reduce(self.shape)
    
    def __contains__(self, val):
        return (val is Missing and self.hasSparse()) \
               or val in self.val
    
    def min(self,skip_sparse=True,out_empty=Missing):
        if(not skip_sparse and self.hasSparse()):
            return out_empty
            
        if(len(self.val) == 0):
            return out_empty
        else:
            return self.val.min()  #missing acts as high value apparently

    def max(self,skip_sparse=True,out_empty=Missing):
        if(not skip_sparse and self.hasSparse()):
            return out_empty
            
        if(len(self.val) == 0):
            return out_empty
        else:
            return self.val.max()  #missing acts as high value apparently

    ##def all(self,skip_sparse=True):
    #    if(not skip_sparse and self.hasSparse()):
    #        return False
    #    return self.val.all() 

    #def any(self,skip_sparse=True):
    #    return self.val.any()

    def ravel(self):
        return PosSparse(self.val, self.sorted_pos, (numpy.multiply.reduce(self.shape),))
    
    def clean(self,dtype=None):
        if(self.val.dtype == object):
            filter = ~numpy.equal(self.val,Missing)
            nval = self.val[filter]
            npos = self.sorted_pos[filter]
        else:
            nval = self.val
            npos = self.sorted_pos
        if(not dtype is None):
            nval = numpy.cast[dtype](nval)

        return PosSparse(nval, npos, self.shape)
        
    def full(self,empty_replace=Missing,otype=None):
        res = numpy.zeros(self.shape,dtype=object)
        res[:] = empty_replace
        res[self.sorted_pos] = self.val
        
        if(not otype is None):
            res = numpy.cast[otype](res)
        return FullSparse(res)


    def __iter__(self):
        if(len(self.shape) == 1):
            return PosSparseSingleIterator(self)
        else:
            return PosSparseMultiIterator(self)

    def sparse_pos_iter(self):
        return zip(self.sorted_pos, self.val)
    
    def sparse_idx_iter(self):
        shape = self.shape
        if(len(shape) > 1):
            indices = [numpy.unravel_index(spos,shape) for spos in self.sorted_pos]
        else:
            indices = self.sorted_pos
        return zip(indices, self.val)
    

    def map(self, func, **kwargs):
        otype = kwargs.get('otype',self.val.dtype)
        if('out_empty' in kwargs):
            out_empty = kwargs['out_empty']
        else:
            out_empty = func(Missing)

        if(otype == object):
            fval = cutils.darray([func(elem) for elem in self.val])
        else:
            func = numpy.frompyfunc(func,1,1)
            fval = numpy.cast[otype](func(self.val))
    
        res = PosSparse(numpy.cast[otype](fval), self.sorted_pos, self._shape)
        if(not out_empty is Missing):
            res = res.full(empty_replace=out_empty)
        else:
            res = res.clean()
        return res

    
    def sparse_filter(self,filter):
        return self.full().sparse_filter(filter) #FIXME make more efficient

    def __getitem__(self, idx):
        if(isinstance(idx,int)):
            if(idx >= self._length):
                raise IndexError, "Index out of bounds: %d >= %d" % (idx, self._length)
            val_idx = numpy.searchsorted(self.sorted_pos, idx)
            if(val_idx < len(self.sorted_pos) and self.sorted_pos[val_idx] == idx):
                return self.val[val_idx]
            else:
                return Missing
        elif(len(self.shape) == 1 and isinstance(idx,slice)):
            idx = numpy.arange(*idx.indices(len(self)))
        elif(len(self.shape) == 1 and operator.isSequenceType(idx) and len(idx) > 0 and not isinstance(idx[0],bool)):
            idx = numpy.cast[int](idx)
            assert not (idx.max() >= numpy.multiply.reduce(self.shape)), "Index(es) out of bounds: %s >= %d" % (str(idx[idx >= self.shape[0]]), self.shape[0])
        else:
            #this is inefficient, but it works
            idx = numpy.arange(numpy.multiply.reduce(self.shape)).reshape(self.shape)[idx]
        nshape = idx.shape
        if(len(self.val) > 0): 
            idx = idx.ravel()
            val_idx = numpy.searchsorted(self.sorted_pos, idx)
            val_idx[val_idx == len(self.val)] = 0

            nval = self.val[val_idx]
            npos = self.sorted_pos[val_idx]
            filter = npos == idx

            nval = nval[filter]
            npos = npos[filter]
        else:
            nval = []
            npos = []

        return PosSparse(nval,npos,nshape)


    def __repr__(self):
        res = []
        for pos,val in self.sparse_idx_iter():
            res.append(str(pos) + ": " + str(val))
        
        res.append(" shape: " + str(self.shape))
        return ", ".join(res)


class PosSparseSingleIterator(object):
    def __init__(self, sparse_list):
        self.sparse_list = sparse_list
        self.pos = 0
        self.sparse_pos = 0
    
    def next(self):
        if(self.pos >= len(self.sparse_list)):
            raise StopIteration
        if(self.sparse_pos < len(self.sparse_list.sorted_pos) and self.sparse_list.sorted_pos[self.sparse_pos] == self.pos):
            res = self.sparse_list.val[self.sparse_pos]
            self.sparse_pos += 1
        else:
            res = Missing
        self.pos += 1
        return res

class PosSparseMultiIterator(object):
    def __init__(self, sparse_array):
        self.rowstep = numpy.multiply.reduce(numpy.array(sparse_array.shape)[1:])
        self.rowidx = numpy.arange(rowstep)
        self.sparse_array = sparse_array
        self.pos = 0
    
    def next(self):
        if(self.pos >= self.sparse_list.shape[0]):
            raise StopIteration
        
        res = self.sparse_array[self.pos * self.rowstep + self.rowidx]
        self.pos += 1
        return res

class FullSparse(numpy.ndarray):
    def __new__(cls,obj):
        res = numpy.array(obj).view(cls)
        return res
    
    def getVal(self):
        return self.view(numpy.ndarray)
    val = property(fget=getVal)
    
    def getSortedPos(self):
        return numpy.arange(len(self))
    sorted_pos = property(fget=getSortedPos)
    
    def _getClasses(self):
        classes = set()
        if(self.val.dtype == object):
            classes |= set([value.__class__ for value in self.ravel()])
        else:
            classes.add(self.val.dtype.type)

        return classes
    classes=property(fget=_getClasses)

    def map(self, func, **kwargs):
        if('out_empty' in kwargs):
            out_empty = kwargs['out_empty']
            if('has_missing' in kwargs):
                func = elem_empty_filter(func,out_empty)
        
        otype = kwargs.get('otype',self.dtype)
        if(otype == object):
            res = cutils.darray([func(elem) for elem in self.ravel()])
            res.shape = self.shape
        else:
            func = numpy.frompyfunc(func,1,1)
            res = numpy.cast[otype](func(self))
        
        return FullSparse(res)

    def sparse_filter(self,filter):
        nval = self.val[filter]
        npos = numpy.where(filter)[0]
        return PosSparse(nval,npos, self.shape).clean()

    def clean(self):
        val = self.ravel()
        filter = ~numpy.equal(val,Missing)
        npos = numpy.where(filter)[0]
        nval = val[filter]
        return PosSparse(nval,npos, self.shape)
       
    def hasSparse(self):
        return numpy.equal(self,Missing).any()

    def full(self,empty_replace=Missing,otype=None):
        if(not empty_replace is Missing):
            missing_filter = numpy.equal(self,Missing)
            res = self.copy()
            res[missing_filter] = empty_replace
        else:
            res = self
        if(not otype is None):
            res = numpy.cast[otype](res)
        return res
    
    def min(self,skip_sparse=True,out_empty=Missing):
        if(len(self) == 0):
            return out_empty

        if(skip_sparse):
            res = self.val.min()  #missing acts as high value apparently
            if(res is Missing):
                res = out_empty
            return res
        else:
            if self.hasSparse():
                return out_empty
            else:
                return self.val.min()

    def max(self,skip_sparse=True,out_empty=Missing):
        if(len(self) == 0):
            return out_empty

        if(skip_sparse):
            filter = ~numpy.equal(self,Missing)
            r = self[filter]
            if(len(r) == 0):
                return out_empty
            else:
                return r.val.max()
        else:
            return self.val.max() #max always returns Missing if available


    #def all(self,skip_sparse=True):
    #    if(skip_sparse):
    #        filter = ~numpy.equal(self,Missing)
    #        return self[filter].val.all()
    #    else:
    #        return self.val.all()

    #def any(self,skip_sparse=True):
    #    return self.val.any()



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


