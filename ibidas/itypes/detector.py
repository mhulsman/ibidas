"""Detector and scanner code. Used to determine data types.

When importing data, types can be represented in multiple ways. 
An integer can be represented as a string, etc. These have 
to be detected (and converted to a standard format). In this module, 
there is a general Detector class, which handles Scanner classes, which
do the actual type checking. 
Main workflow:

d = Detector()
d.process(<your data obj>)
[d.process(<your data obj>)]
d.getType()

Helper classes
--------------
TypeScanner (derivatives): Scan for specific types.
DimRep:                    Process and store dimension shapes
DimensionEqualizer:        Find common dimensions


Overview
--------------------------
Scanners follow a hierarchy, starting with the AnyScanner, which has
children such as CollectionScanner, TupleScanner and NumberScanner. 
Each child can have children of his own. 

Each detector first performs a depth-first search of allowable scanners,
and then during data processing goes up in the hiearchy if the more specific
children-scanners cannot match the data anymore.

Scanners implement a 'scan' method, which gets a sequence of objects, and
should return False/True if these are objects are acceptable. Detector performs
a common detection step in that it determines a set of class types available in the
sequence. 

Once the user ask for the type, the detector asks every active scanner for a type, and
next will (try) to determine which type should be returned. For this, the scanner class
should have a getType method, where it creates an type based on the collected information.

During the data scanning, scanners will sometimes not be able to match the data, after
which their parent scanner is activated. Earlier data is then not scanned again, instead
the child scanner should transfer its collected information to the parent scanner during
its own unregistration (the 'unregister' method will be called with create_parent flag set to 
True). 

If a scanner has to scan subtypes, it should open a subdetector class (for which one 
can use the getSubDetector method implemented in typescanner). This detector should than be 
feeded the subobjects. 

To determine dimensions, one can use the DimensionEqualizer/DimRep classes. Each dimension 
should have its own DimRep class, which can process dimensions lengths using the processLengths
method. It will store if this length is fixed/variable/repetitive, and allow for comparison between
dimensions. One can open a new DimRep class with getDimRep method. If during scanning it becomes clear
that the number of dimensions has to be reduced, one can do this with the reduceDimReps method. 

During the call to getType, DimensionEqualizer will compare DimReps, and assign dimension objects
to each. These can be obtained from the .dim attributed during type creation. 


Adding a new scanner class
--------------------------
The scanner hierarchy is separate from the class hierarchy. So one should find the class which has 
the most common implementation (often just TypeScanner).
Next, one should implement the scan method. See for examples below.
The parentcls attribute should be used to determine the parent scanner,
the registerTypeScanner method to register the scanner. 
"""

import sparse_arrays
import numpy
import array
import rtypes
from rtypes import Missing, DATA_NORMAL, DATA_INPUT, DATA_FROZEN
from collections import defaultdict
import cutils
import operator
import util
import dimensions
import convertors

_scanchildren = defaultdict(list)
def registerTypeScanner(newscancls):
    _scanchildren[newscancls.parentcls].append(newscancls)

missing_cls = set([None.__class__, rtypes.MissingType])
missing_val = set([None, Missing])


class Detector(object):
    def __init__(self, parent_scanner=None, dim_eq=None):
        self._scanners = None
        self.objectclss = set()
        self.count_elem = 0

        if not dim_eq:
            dim_eq = DimEqualizer()

        self.dim_eq = dim_eq
        self.parent_scanner = parent_scanner
        return None

    def process(self, obj):
        self.processSeq(sparse_arrays.FullSparse(cutils.darray([obj])))

    def processSeq(self, seq):
        if not isinstance(seq, (sparse_arrays.FullSparse, sparse_arrays.PosSparse)):
            if isinstance(seq, (set, frozenset)):
                seq = cutils.darray(list(seq))
            elif not isinstance(seq, numpy.ndarray):
                seq = cutils.darray(seq)
            seq = sparse_arrays.FullSparse(seq)
        
        objclasses = seq.classes
        self.objectclss |= objclasses
        elem_count = len(seq)

        if not objclasses.issubset(missing_cls):
            if not self._scanners:
                test_seq = seq[:100]
                seq = seq[100:]
                self._scanners = list(self._findAcceptableDescendants(test_seq))
            if len(seq) > 0:
                for pos in xrange(len(self._scanners) - 1, -1, -1):
                    scanner = self._scanners[pos]
                    while not scanner.scan(seq):
                        if not any([scanner.parentcls in s.ancestor_classes for s in self._scanners]):
                            scanner = scanner.unregister(create_parent=True)
                            self._scanners[pos] = scanner
                            continue
                        scanner.unregister()
                        del self._scanners[pos]
                        break
        self.count_elem += elem_count

    def getType(self):
        if not self._scanners:
            return rtypes.TypeUnknown()
        if not self.parent_scanner:
            self.dim_eq.processDims()

        restypes = [scanner.getType() for scanner in self._scanners]
        
        restypes = [t for t in restypes if t is not None]
        
        
        assert not len(restypes) == 0,'BUG: no valid type could be detected'

        if len(restypes) > 1:
            #FIXME
            if len(restypes) > 1:
                if len(restypes) > 1:
                    if len(restypes) > 1:
                        raise RuntimeError, 'Could not decide between types: ' + str(restypes)
        return restypes[0]

    def _findAcceptableDescendants(self, seq, scanner=None):
        schildren = [sc(self) for sc in _scanchildren[scanner.__class__]]
       
        res = set()
        for sc in schildren:
            if sc.scan(seq):
                res.update(self._findAcceptableDescendants(seq, sc))
            else:
                sc.unregister()
        
        if not res:
            return (scanner,)

        if scanner:
            scanner.unregister()

        return res

    def hasMissing(self):
        return not missing_cls.isdisjoint(self.objectclss)



LENGTH_NOINIT = 0    #length type not yet initialized
LENGTH_FIXED = 1     #lengths are fixed (all the same or missing)
LENGTH_VAR = 2       #variable lengths
LENGTH_REPEAT = 3    #lengths repeated in a pattern dictated by parent dimensions
class DimRep(object):
    def __init__(self, parent_scanner, index=0):
        self.parent_scanner = parent_scanner
        self.index = index

        self.lengths = None        #stores length data. FIXED: int, VAR: list of lengths, REPEAT: list of lengths
        self.length_count = 0      #FIXED: stores number of elements, REPEAT: number of repeats
        self.length_type = LENGTH_NOINIT

        self.nparents = 0

        self.has_missing = False
        self.dim = None
        self.dirty = True

    def processLengths(self, lengths, has_missing=False):
        self.has_missing = max(self.has_missing, has_missing)
        self.dirty = True

        if has_missing and lengths.dtype == object:
            lengths = lengths.full(empty_replace=-1, otype=int)

        if self.length_type == LENGTH_FIXED or self.length_type == LENGTH_NOINIT:
            lenset = set(lengths)
            lenset.discard(-1)
            if len(lenset) == 0:
                self.length_count += len(lengths)
            elif len(lenset) > 1:
                self._fullLengths()
                self.processLengths(lengths)
            else:
                length = lenset.pop()
                if self.length_type == LENGTH_NOINIT:
                    self.lengths = length
                    self.length_type = LENGTH_FIXED
                if self.lengths == length:
                    self.length_count += len(lengths)
                else:
                    self._fullLengths()
                    self.processLengths(lengths)
        else:
            if self.length_type == LENGTH_VAR:
                self.lengths.extend(lengths)
            else:
                self._fullLengths()
                self.processLengths(lengths)

    def _fullLengths(self):
        if self.length_type == LENGTH_FIXED:
            self.lengths = [self.lengths] * self.length_count
        elif self.length_type == LENGTH_NOINIT:
            self.lengths = []
        elif self.length_type == LENGTH_REPEAT:
            self.lengths = self.lengths * self.length_count
            self.nparents = 0
        self.length_type = LENGTH_VAR

    def _setDim(self, dim):
        self.dirty = False
        self.dim = dim

    def checkRepeats(self, repeat_length, nparents):
        if self.length_type == LENGTH_FIXED or self.length_type == LENGTH_NOINIT:
            return False

        if self.length_type == LENGTH_REPEAT:
            return self.lengths == repeat_length

        lengths = self.lengths
        if len(lengths) % repeat_length:
            return False

        repeat = lengths[:repeat_length]
        lengths = lengths[repeat_length:]
        while lengths:
            if repeat != lengths[:repeat_length]:
                return False
            lengths = lengths[repeat_length:]
        self.length_count = len(lengths) / repeat_length
        self.lengths = repeat
        self.nparents = nparents
        return True

    def getDim(self):
        assert self.dirty == False,'DimRep is dirty, has dim equalizer been processed?'
        return self.dim

    def combineWith(self, other):
        self.dirty = True
        if self.length_type == LENGTH_FIXED and other.length_type == LENGTH_FIXED:
            assert self.length_count == other.length_count,'Cannot combine dimreps with unequal lengths'
            self.lengths = self.lengths * other.lengths
        else:
            self._fullLengths()
            other._fullLengths()
            assert len(self.lengths) == len(other.lengths),'Cannot combine dimreps with unequal lenghts'
            self.lengths = list(numpy.multiply(self.lengths, other.lengths))

    def getParentDimReps(self):
        return self.parent_scanner.getDimReps(self.index)



class DimEqualizer(object):

    def __init__(self):
        self.dimreps = []

    def registerDimRep(self, dimrep):
        self.dimreps.append(dimrep)

    def unregisterDimRep(self, dimrep):
        del self.dimreps[self.dimreps.index(dimrep)]

    def _processParents(self, dimrep):
        cur_parents = dimrep.getParentDimReps()
        for parent in cur_parents:
            if parent.dirty:
                self._attachDim(parent)
        
        pos = len(cur_parents)
        repeat_length = 1
        while pos:
            pos -= 1
            parent = cur_parents[pos]
            assert (parent.length_type != LENGTH_NOINIT), 'Unitialized parent should not be possible'
            
            if parent.length_type == LENGTH_VAR: #A VAR as parent means that no repeat could be found, i.e. lengths vary across all parents
                break
            elif parent.length_type == LENGTH_REPEAT:
                repeat_length *= sum(parent.lengths)
                pos -= parent.nparents
            else:
                repeat_length *= parent.lengths

            if dimrep.checkRepeats(repeat_length, len(cur_parents) - pos):
                cur_parents = cur_parents[-dimrep.nparents:]
                break
        return cur_parents

    def _attachDim(self, dimrep):
        assert (dimrep.length_type != LENGTH_NOINIT), 'Unitialized parent should not be possible'

        if dimrep.length_type != LENGTH_FIXED:
            cur_parents = self._processParents(dimrep)
            nparents = max(len(cur_parents), 1)
        else:
            nparents = 0

        for dr in self.dimreps:
            if dr.dirty is True:
                continue
            if dr.has_missing != dimrep.has_missing:
                continue
            if dimrep.length_type != dr.length_type:
                continue
            if dr.lengths != dimrep.lengths:
                continue
            if dimrep.length_type == LENGTH_FIXED:
                dimrep._setDim(dr.dim)
                return dr.dim
            match_parents = dr.getParentDimReps()[-nparents:]
            if len(match_parents) != len(cur_parents):
                continue
            for mp, cp in zip(match_parents, cur_parents):
                if mp.dirty:
                    self._attachDim(mp)
                if cp.dim is not mp.dim:
                    break
            else:
                dimrep._setDim(dr.dim)
                return dr.dim
        if dimrep.length_type == LENGTH_FIXED:
            ndim = dimensions.Dim(dimrep.lengths, nparents, dimrep.has_missing)
        else:
            ndim = dimensions.Dim(dimensions.UNDEFINED, nparents, dimrep.has_missing)
        dimrep._setDim(ndim)
        return ndim

    def processDims(self):
        for dr in self.dimreps:
            if dr.dirty:
                self._attachDim(dr)



class TypeScanner(object):
    typecls = rtypes.TypeAny
    good_cls = set()
    convertor=convertors.BaseConvertor

    def __init__(self, detector):
        self.detector = detector
        self.dimreps = []

    def unregister(self, create_parent=False):
        for d in self.dimreps:
            self.detector.dim_eq.unregisterDimRep(d)

        if create_parent:
            return self.parentcls(self.detector)

    def scan(self, seq):
        return self.detector.objectclss.issubset(self.good_cls)

    def getType(self):
        cv = self.convertor(self.detector.objectclss.copy()) 
        return self.typecls(self.detector.hasMissing(), convertor=cv, data_state=DATA_INPUT)

    def getAncestorScanners(self):
        if not hasattr(self.__class__, 'ancest_cls_cache'):
            acc = []
            cur_cls = self.__class__.parentcls
            while cur_cls is not None.__class__:
                acc.append(cur_cls)
                cur_cls = cur_cls.parentcls
            self.__class__.ancest_cls_cache = acc
        return self.__class__.ancest_cls_cache
    ancestor_scanners = property(fget=getAncestorScanners)

    def getSubDetector(self, id=0):
        id = 'subdetector_' + str(id)
        if id not in self.__dict__:
            d = Detector(self, self.detector.dim_eq)
            setattr(self, id, d)
            if self.detector.count_elem:
                d.processSeq(sparse_arrays.PosSparse([], shape=(self.detector.count_elem,)))
        return getattr(self, id)

    def getDimReps(self, last_index=None):
        ps = self.detector.parent_scanner
        if last_index is None:
            sdimreps = self.dimreps[:]
        else:
            sdimreps = self.dimreps[:last_index]
        if ps:
            if not self.dimreps:
                return ps.getDimReps()
            res = ps.getDimReps()
            res.extend(sdimreps)
            return res
        return sdimreps

    def getDimRep(self, i):
        if len(self.dimreps) <= i:
            assert len(self.dimreps) == i,'DimReps not requested in order'
            d = DimRep(self, i)
            if self.detector.count_elem:
                d.processLengths(sparse_arrays.PosSparse([], shape=(self.detector.count_elem,)), has_missing=True)
            self.detector.dim_eq.registerDimRep(d)
            self.dimreps.append(d)
        return self.dimreps[i]

    def reduceDimReps(self, i):
        while len(self.dimreps) > i:
            d = self.dimreps.pop()
            self.dimreps[-1].combineWith(d)
            self.detector.dim_eq.unregisterDimRep(d)
        self.min_dim = i



class AnyScanner(TypeScanner):
    parentcls = None.__class__

    def unregister(self, create_parent=False):
        if create_parent:
            raise RuntimeError, 'Attempting to find parent for AnyScanner'
        return TypeScanner.__init__(self,create_parent)

    def scan(self, seq):
        return True
registerTypeScanner(AnyScanner)
TypeScanner.parentcls = AnyScanner


class TupleScanner(TypeScanner):
    good_cls = set([tuple, None.__class__, rtypes.MissingType])

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.max_len = 0
        self.min_len = 999999999

    def getType(self):
        fieldnames = ['f' + str(i) for i in xrange(self.max_len)]
        subtypes = [self.getSubDetector(i).getType() for i in xrange(self.max_len)]
        cv = self.convertor(self.detector.objectclss.copy()) 
        return rtypes.TypeTuple(self.detector.hasMissing(), subtypes, fieldnames, convertor=cv, data_state=DATA_INPUT)

    
    def scan(self, seq):
        if not self.detector.objectclss.issubset(self.good_cls):
            return False

        l = seq.map(len, otype=int, out_empty=0, has_missing=self.detector.hasMissing())

        maxlen = l.max(out_empty=0)
        minlen = l.min(out_empty=self.min_len)
        self.max_len = max(maxlen, self.max_len)
        self.min_len = min(minlen, self.min_len)

        for i in xrange(self.max_len):
            d = self.getSubDetector(i)
            f = operator.itemgetter(i)
            if i < self.min_len:
                subseq = seq.map(f, otype=object, out_empty=Missing)
            else:
                subseq = seq.sparse_filter(l > i).map(f, out_empty=Missing, otype=object, has_missing=True)
            d.processSeq(subseq)
        return True
registerTypeScanner(TupleScanner)


class NamedTupleScanner(TypeScanner):
    bad_cls = set([tuple])
    convertor=convertors.NamedTupleConvertor

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.names = set()

    def getType(self):
        fieldnames = [name for name in self.names]
        subtypes = [self.getSubDetector(name).getType() for name in self.names]
        cv = self.convertor(self.detector.objectclss.copy()) 
        return rtypes.TypeTuple(self.detector.hasMissing(), subtypes, fieldnames, convertor=cv, data_state=DATA_INPUT)

    def scan(self, seq):
        if self.bad_cls.issubset(self.detector.objectclss):
            return False

        for cls in self.detector.objectclss:
            if cls is None.__class__ or cls is rtypes.MissingType:
                continue
            if not(issubclass(cls, tuple) and hasattr(cls, '_fields')):
                return False
            self.names.update(cls._fields)

        for name in self.names:
            d = self.getSubDetector(name)
            def getname(elem):
                try:
                    return getattr(elem,name)
                except AttributeError:
                    return Missing
            subseq = seq.map(getname, otype=object)
            d.processSeq(subseq)
        return True
registerTypeScanner(NamedTupleScanner)

class DictScanner(TypeScanner):
    good_cls = set([dict, None.__class__, rtypes.MissingType])
    convertor=convertors.DictConvertor

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.names = set()

    def getType(self):
        fieldnames = [name for name in self.names]
        subtypes = [self.getSubDetector(name).getType() for name in self.names]
        cv = self.convertor(self.detector.objectclss.copy()) 
        return rtypes.TypeTuple(self.detector.hasMissing(), subtypes, fieldnames, convertor=cv, data_state=DATA_INPUT)

    def scan(self, seq):
        if not self.detector.objectclss.issubset(self.good_cls):
            return False

        for elem in seq:
            if not (elem is Missing or elem is None):
                self.names.update(elem.keys())

        for name in self.names:
            d = self.getSubDetector(name)
            f = operator.itemgetter(name)
            def getname(elem):
                try:
                    return elem[name]
                except KeyError:
                    return Missing
            subseq = seq.map(getname, otype=object)
            d.processSeq(subseq)
        return True
registerTypeScanner(DictScanner)


class ContainerScanner(TypeScanner):
    good_cls = set([set, frozenset, None.__class__, rtypes.MissingType, list, array.array, numpy.ndarray, sparse_arrays.FullSparse, sparse_arrays.PosSparse])
    bad_cls = set([tuple, str])

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.min_dim = None

    def getType(self):
        subtype = self.getSubDetector().getType()
        
        dims = [self.getDimRep(i).dim for i in xrange(self.min_dim)]
        cv = self.convertor(self.detector.objectclss.copy()) 
        return rtypes.TypeArray(self.detector.hasMissing(), dims, (subtype,), convertor=cv, data_state=DATA_INPUT)

    def scan(self, seq):
        if not self.detector.objectclss.issubset(self.good_cls):
            has_missing = self.detector.hasMissing()
            if not self.detector.objectclss.isdisjoint(self.bad_cls):
                return False

            for cls in self.detector.objectclss:
                if issubclass(cls, tuple):
                    return False

            l = seq.map(operator.isSequenceType, otype=bool, out_empty=True, has_missing=has_missing)
            if not l.all():
                return False
        if(self.min_dim == 1):
            dr = self.getDimRep(0)
            nelems = seq.map(getnelem, otype=object, out_empty=Missing, has_missing=has_missing)
            dr.processLengths(nelems, has_missing=has_missing)
        else:
            shapes = seq.map(getshape, otype=object, out_empty=Missing, has_missing=has_missing)
            shapelens = shapes.map(len, otype=object, out_empty=Missing, has_missing=has_missing)
            min_dim = shapelens.min()
            max_dim = shapelens.max()

            if self.min_dim is None:
                self.min_dim = min_dim
            else:
                if min_dim < self.min_dim:
                    self.reduceDimReps(min_dim)

            for i in xrange(self.min_dim):
                dr = self.getDimRep(i)
                if i == self.min_dim and self.min_dim < max_dim:
                    red = numpy.multiply.reduce
                    def reduceshape(shape):
                        return red(shape[i:])
                    f = reduceshape
                else:
                    f = operator.itemgetter(i)
                nelems = shapes.map(f, otype=object, out_empty=Missing, has_missing=has_missing)
                dr.processLengths(nelems, has_missing=has_missing)
        
        d = self.getSubDetector()
        for subseq in seq.ravel():
            if not (subseq is Missing or subseq is None):
                d.processSeq(subseq)
        return True

def getshape(elem):
    try:
        return elem.shape
    except: 
        return (len(elem),)

def getnelem(elem):
    try:
        return len(elem.ravel())
    except:
        return len(elem)
registerTypeScanner(ContainerScanner)


class SetScanner(ContainerScanner):
    parentcls = ContainerScanner
    good_cls = set([set, frozenset, None.__class__, rtypes.MissingType])
    convertor=convertors.SetConvertor

    def getType(self):
        subtype = self.getSubDetector().getType()
        dims = (self.getDimRep(0).dim,)
        cv = self.convertor(self.detector.objectclss.copy()) 
        return rtypes.TypeSet(self.detector.hasMissing(), dims, (subtype,), convertor=cv, data_state=DATA_INPUT)

    def unregister(self, create_parent=False):
        parent = ContainerScanner.unregister(self, create_parent)
        if create_parent:
            parent.min_dim = 1
            parent.dimreps = self.dimreps
        return parent

    def scan(self, seq):
        has_missing = self.detector.hasMissing()
        if not self.detector.objectclss.issubset(self.good_cls):
            return False
        
        dr = self.getDimRep(0)
        nelems = seq.map(len, otype=object, out_empty=Missing, has_missing=has_missing)
        dr.processLengths(nelems, has_missing=has_missing)
        d = self.getSubDetector()
        for subseq in seq.ravel():
            if not (subseq is Missing or subseq is None):
                d.processSeq(subseq)
        return True
registerTypeScanner(SetScanner)


class StringScanner(TypeScanner):
    good_cls = set([str, unicode, None.__class__, rtypes.MissingType, numpy.string_, numpy.unicode_])
    unicode_cls = set([unicode, numpy.unicode_])
    convertor=convertors.StringConvertor

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.max_nchars = 0

    def getType(self):
        if self.unicode_cls.isdisjoint(self.detector.objectclss):
            ntype = rtypes.TypeBytes
        else:
            ntype = rtypes.TypeString

        if self.max_nchars < 32:
            d = dimensions.Dim(self.max_nchars, 0, self.detector.hasMissing())
        else:
            d = dimensions.Dim(dimensions.UNDEFINED, len(self.getDimReps(0)), self.detector.hasMissing())

        dims = (d,)
        cv = self.convertor(self.detector.objectclss.copy()) 
        return ntype(self.detector.hasMissing(), dims, convertor=cv, data_state=DATA_INPUT)

    def scan(self, seq):
        has_missing = self.detector.hasMissing()
        if not self.detector.objectclss.issubset(self.good_cls):
            return False
        max_nchars = seq.map(len, otype=int, out_empty=-1, has_missing=has_missing).max()
        self.max_nchars = max(self.max_nchars, max_nchars)
        return True
registerTypeScanner(StringScanner)


class NumberScanner(TypeScanner):
    __doc__ = 'Number scanner, accepts all number objects.'
    typecls = rtypes.TypeNumber
    good_cls = set((bool, float, complex, long, int, None.__class__, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float32, numpy.float64, numpy.complex64, numpy.complex128, numpy.bool_, rtypes.MissingType))

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.has_nan = False

    def getType(self):
        if not self.detector.hasMissing():
            pass
        has_missing = self.detector.hasMissing()
        cv = self.convertor(self.detector.objectclss.copy()) 

        return self.typecls(has_missing, convert_type=self.__class__, convertor=cv, data_state=DATA_INPUT)

    def scan(self, seq):
        if not self.detector.objectclss.issubset(self.good_cls):
            return False
        if not self.has_nan:
            self.has_nan = numpy.nan in seq
        return True
registerTypeScanner(NumberScanner)


class RealScanner(NumberScanner):
    __doc__ = 'Real scanner, accepts all real and integer objects.'
    parentcls = NumberScanner
    typecls = rtypes.TypeReal64
    good_cls = set((float, bool, None.__class__, numpy.float32, numpy.float64, rtypes.MissingType, long, int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.bool_))
registerTypeScanner(RealScanner)


class IntegerScanner(TypeScanner):
    parentcls = RealScanner
    ustepvals = [256, 65536, 4294967296L, 18446744073709551616L]
    uinttypes = [rtypes.TypeUInt8, rtypes.TypeUInt16, rtypes.TypeUInt32, rtypes.TypeUInt64]
    istepvals = [128, 32768, 2147483648L, 9223372036854775808L]
    inttypes = [rtypes.TypeInt8, rtypes.TypeInt16, rtypes.TypeInt32, rtypes.TypeInt64]
    
    good_cls = set((bool, long, int, None.__class__, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.bool_, rtypes.MissingType))
    
    numpy_minmax = {numpy.dtype('bool'): (0, 1), numpy.dtype('int8'): (-128, 127), numpy.dtype('uint8'): (0, 255), numpy.dtype('int16'): (-32768, 32767), numpy.dtype('uint16'): (0, 65535), numpy.dtype('int32'): (-2147483648, 2147483647), numpy.dtype('uint32'): (0, 4294967295L), numpy.dtype('int64'): (-9223372036854775808L, 9223372036854775807L), numpy.dtype('uint64'): (0, 18446744073709551615L)}
    array_minmax = {'b': (-128, 127), 'B': (0, 255), 'h': (-32768, 32767), 'H': (0, 65535), 'i': (-2147483648, 2147483647), 'I': (0, 4294967295L), 'l': (-2147483648, 2147483647), 'L': (0, 4294967295L)}

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.max_val = 0
        self.min_val = 0

    def getType(self):
        out_type = rtypes.TypeInteger
        if self.min_val >= 0:
            for stepval, rtype in zip(self.ustepvals, self.uinttypes):
                if self.max_val < stepval:
                    out_type = rtype
                    break
        else:
            for stepval, rtype in zip(self.istepvals, self.inttypes):
                if -self.min_val <= stepval and self.max_val < stepval:
                    out_type = rtype
                    break
        cv = self.convertor(self.detector.objectclss.copy()) 
        return out_type(self.detector.hasMissing(), convertor=cv, data_state=DATA_INPUT)

    def scan(self, seq):
        if not self.detector.objectclss.issubset(self.good_cls):
            return False
        
        if isinstance(seq, numpy.ndarray) and not seq.dtype == object:
            minmax = self.numpy_minmax[seq.dtype]
        elif isinstance(seq, array.array):
            minmax = self.array_minmax[seq.typecode]
        elif len(seq) == 0:
            minmax = (0, 0)
        else:
            minmax = (seq.min(out_empty=0), seq.max(out_empty=0))

        self.min_val = min(minmax[0], self.min_val)
        self.max_val = max(minmax[1], self.max_val)
        return True
registerTypeScanner(IntegerScanner)


class BoolScanner(TypeScanner):
    parentcls = IntegerScanner
    typecls = rtypes.TypeBool
    good_cls = set((numpy.bool_, bool, None.__class__, rtypes.MissingType))
registerTypeScanner(BoolScanner)



class StringRealScanner(RealScanner):
    good_cls = set([str, unicode, None.__class__, rtypes.MissingType, numpy.string_, numpy.unicode_])
    parentcls = StringScanner
    convertor=convertors.StringFloatConvertor

    def scan(self, seq):
        has_missing = self.detector.hasMissing()
        if not self.detector.objectclss.issubset(self.good_cls):
            return False
        try:
            if has_missing:
                float_vals = seq.map(int, otype=object, out_empty=Missing, has_missing=has_missing)
            else:
                float_vals = seq.map(int, otype=float, has_missing=has_missing)
        except ValueError:
            return False
        return RealScanner.scan(self, float_vals)
registerTypeScanner(StringRealScanner)


class StringIntegerScanner(IntegerScanner):
    good_cls = set([str, unicode, None.__class__, rtypes.MissingType, numpy.string_, numpy.unicode_])
    parentcls = StringRealScanner
    convertor=convertors.StringIntegerConvertor

    def scan(self, seq):
        has_missing = self.detector.hasMissing()
        if not self.detector.objectclss.issubset(self.good_cls):
            return False
        try:
            if has_missing:
                int_vals = seq.map(int, otype=object, out_empty=Missing, has_missing=has_missing)
            else:
                int_vals = seq.map(int, otype=int, has_missing=has_missing)
        except ValueError:
            return False
        return IntegerScanner.scan(self, int_vals)
registerTypeScanner(StringIntegerScanner)
