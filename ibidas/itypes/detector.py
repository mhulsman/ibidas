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

.. rubric:: Helper classes
    TypeScanner (derivatives): Scan for specific types.
    DimRep:                    Process and store dimension shapes
    DimensionEqualizer:        Find common dimensions


.. rubric:: Overview
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

    Once the user asks for the type, the detector asks every active scanner for a type, and
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


.. rubric:: Adding a new scanner class
    The scanner hierarchy is separate from the class hierarchy. So one should find the class which has 
    the most common implementation (often just TypeScanner).
    Next, one should implement the scan method. See for examples below.
    The parentcls attribute should be used to determine the parent scanner,
    the registerTypeScanner method to register the scanner. 
"""

import numpy
import array
from collections import defaultdict
import operator
import platform
from itertools import chain

import rtypes
from ..constants import *
from ..utils import sparse_arrays, module_types
from ..utils.missing import *

_delay_import_(globals(),"dimensions")
_delay_import_(globals(),"dimpaths")
_delay_import_(globals(),"..utils","cutils","util")

_scanchildren = defaultdict(list)
def registerTypeScanner(newscancls):
    _scanchildren[newscancls.parentcls].append(newscancls)

missing_cls = set([MissingType])
missing_val = set([Missing])


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

    def setParentDimensions(self,dims):
        assert self.count_elem == 0, "Cannot perform setParentDimensions after process(Seq) has been called"
        self._scanners = [OuterContainerScanner(self,dims)]

    def process(self, obj):
        self.processSeq(sparse_arrays.FullSparse(util.darray([obj])))

    def processSeq(self, seq):
        if not isinstance(seq, (sparse_arrays.FullSparse)):
            if isinstance(seq, (set, frozenset)):
                seq = util.darray(list(seq))
            elif not isinstance(seq, numpy.ndarray):
                seq = util.darray(seq)
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
                        del self._scanners[pos]
                        if not any([scanner.parentcls in s.ancestor_scanners for s in self._scanners]):
                            scanner = scanner.unregister(create_parent=True)
                            self._scanners.insert(pos,scanner)
                            continue
                        scanner.unregister()
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

        if has_missing:
            lenghts = lengths.replace_missing(-1, otype=int)

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
        self.length_type = LENGTH_REPEAT
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


class FixedDimRep(DimRep):
    def __init__(self, parent_scanner, index, dim):
        self.parent_scanner = parent_scanner
        self.index = index

        if(dim.isVariable()):
            self.length_type = LENGTH_VAR
            self.lengths = [] 
        else:
            self.length_type = LENGTH_FIXED
            self.lengths = dim.shape
            #FIXME: give lengths of variable dimensions from data, will also enable repeat var dimensions
        
        self.nparents = 0
        self.has_missing = dim.has_missing
        self.dim = dim
        self.dirty = False


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
            elif parent.lengths == UNDEFINED: # length is UNDEFINED, no further checks possible
                break
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
            parents = cur_parents
        else:
            parents = []

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
            match_parents = dr.getParentDimReps()[-len(parents):]
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
            ndim = dimensions.Dim(dimrep.lengths, tuple(), dimrep.has_missing,name="d"+str(self.dimid()))
        else:
            ndim = dimensions.Dim(UNDEFINED, (True,) * len(parents), dimrep.has_missing,name="d"+str(self.dimid()))
        dimrep._setDim(ndim)
        return ndim

    def processDims(self):
        self.dimid = util.seqgen().next
        for dr in self.dimreps:
            if dr.dirty:
                self._attachDim(dr)



class TypeScanner(object):
    typecls = rtypes.TypeAny
    good_cls = set()

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
        return self.typecls(self.detector.hasMissing())

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
                d.processSeq([Missing] * self.detector.count_elem)
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
                d.processLengths(sparse_arrays.FullSparse([Missing] * self.detector.count_elem), has_missing=True)
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
    good_cls = set([tuple, MissingType])

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.max_len = 0
        self.min_len = 999999999
        self.fieldnames = []

    def getType(self):
        if(not len(self.fieldnames) == self.max_len):
            self.fieldnames = ['f' + str(i) for i in xrange(self.max_len)]
        subtypes = tuple([self.getSubDetector(i).getType() for i in xrange(self.max_len)])
        return rtypes.TypeTuple(self.detector.hasMissing(), subtypes, self.fieldnames)
    
    def scan(self, seq):
        if not self.detector.objectclss.issubset(self.good_cls):
            return False

        l = seq.map(len, out_empty=0, has_missing=self.detector.hasMissing(), otype=int)

        maxlen = l.max(out_empty=0)
        minlen = l.min(out_empty=self.min_len)
        self.max_len = max(maxlen, self.max_len)
        self.min_len = min(minlen, self.min_len)

        if(len(seq) == 1 and self.max_len == self.min_len):           
            self.fieldnames = util.find_names(seq[0])
            
        for i in xrange(self.max_len):
            d = self.getSubDetector(i)
            f = operator.itemgetter(i)
            if i < self.min_len:
                subseq = seq.map(f, has_missing=self.detector.hasMissing())
            else:
                subseq = seq.filter_tomissing(l > i).map(f, out_empty=Missing, otype=object, has_missing=True)
            d.processSeq(subseq)
        return True
registerTypeScanner(TupleScanner)


class NamedTupleScanner(TypeScanner):
    bad_cls = set([tuple])

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.tuple_cls = None

    def getType(self):
        fieldnames = [util.valid_name(name) for name in self.tuple_cls._fields]
        subtypes = tuple([self.getSubDetector(pos).getType() for pos in range(len(fieldnames))])
        return rtypes.TypeTuple(self.detector.hasMissing(), subtypes, fieldnames)

    def scan(self, seq):
        if self.bad_cls.issubset(self.detector.objectclss):
            return False
        for cls in self.detector.objectclss:
            if cls is MissingType:
                continue
            if not self.tuple_cls:
                if not hasattr(cls, '_fields'):
                    return False
                self.tuple_cls = cls
            elif not self.tuple_cls is cls:
                return False

        fieldlen = len(self.tuple_cls._fields)
        for i in xrange(fieldlen):
            d = self.getSubDetector(i)
            f = operator.itemgetter(i)
            subseq = seq.map(f, has_missing=self.detector.hasMissing())
            d.processSeq(subseq)
        return True
registerTypeScanner(NamedTupleScanner)

class RecordDictScanner(TypeScanner):
    good_cls = set([dict, module_types.soap_struct, MissingType])

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.names = set()

    def getType(self):
        fieldnames = [name for name in self.names]
        subtypes = tuple([self.getSubDetector(name).getType() for name in self.names])
        return rtypes.TypeRecordDict(self.detector.hasMissing(), subtypes, fieldnames)

    def scan(self, seq):
        if not self.detector.objectclss.issubset(self.good_cls):
            return False
        
        names = self.names.copy()
        if dict in self.detector.objectclss:
            assert not module_types.soap_struct in self.detector.objectclss, "dict cannot be mixed with SOAPpy struct type"
            for elem in seq:
                if not (elem is Missing):
                    names.update(elem.keys())
        else:
            assert not dict in self.detector.objectclss, "dict cannot be mixed with SOAPpy struct type"
            for elem in seq:
                if not (elem is Missing):
                    names.update(elem._keys())

        newnames = names - self.names
        for name in newnames:
            if not isinstance(name, basestring) or util.valid_name(name) != name:
                return False
        self.names = names

        if(len(self.names) > 100):
            return False

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
registerTypeScanner(RecordDictScanner)


class ContainerScanner(TypeScanner):
    good_cls = set([set, frozenset, MissingType, list, array.array, numpy.ndarray])
    bad_cls = set([tuple, str, unicode, numpy.unicode_, numpy.string_, module_types.soap_struct])

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.min_dim = None

    def getType(self):
        subtype = self.getSubDetector().getType()
        dims = dimpaths.DimPath(*[self.getDimRep(i).dim for i in xrange(self.min_dim)])
        if self.detector.hasMissing():
            dims[0].has_missing = True

        return dimpaths.dimsToArrays(dims, subtype)

    def scan(self, seq):
        has_missing = self.detector.hasMissing()
        if not self.detector.objectclss.issubset(self.good_cls):
            if not self.detector.objectclss.isdisjoint(self.bad_cls):
                return False

            for cls in self.detector.objectclss:
                if issubclass(cls, tuple):
                    return False

            l = seq.map(operator.isSequenceType, otype=bool, out_empty=True, has_missing=has_missing)
            if not l.all(has_missing=False):
                return False
        
        if(self.min_dim == 1):
            dr = self.getDimRep(0)
            nelems = seq.map(getnelem, otype=object, out_empty=Missing, has_missing=has_missing)
            dr.processLengths(nelems, has_missing=has_missing)
        else:
            shapes = seq.map(getshape, otype=object, out_empty=Missing, has_missing=has_missing)
            shapelens = shapes.map(len, otype=object, out_empty=Missing, has_missing=has_missing)
            min_dim = shapelens.min(has_missing=has_missing)
            max_dim = shapelens.max(has_missing=has_missing)

            if(min_dim is Missing or max_dim is Missing):
                if(self.min_dim):
                    for i in xrange(self.min_dim):
                        dr = self.getDimRep(i)
                        dr.processLengths(shapelens, has_missing=has_missing)
            else:
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
                    dr.processLengths(nelems.ravel(), has_missing=has_missing)
        
        d = self.getSubDetector()
        if(self.min_dim == 1 and not has_missing):
            d.processSeq(list(chain(*seq.ravel())))
        else: 
            for subseq in seq.ravel():
                if not (subseq is Missing):
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


class OuterContainerScanner(ContainerScanner):
    parentcls = None.__class__

    def __init__(self, detector, dims):
        ContainerScanner.__init__(self,detector)
        self.dimreps = [FixedDimRep(self,pos,dim) for pos,dim in enumerate(dims)]

    def scan(self,seq):
        d = self.getSubDetector()
        d.processSeq(seq)
        return True

    def getType(self):
        return self.getSubDetector().getType()


class SetScanner(ContainerScanner):
    parentcls = ContainerScanner
    good_cls = set([set, frozenset, MissingType])

    def getType(self):
        subtype = self.getSubDetector().getType()
        dim = dimensions.Dim(UNDEFINED, (True,) * len(self.getDimReps(0)), self.detector.hasMissing())
        dims = dimpaths.DimPath(dim)
        return rtypes.TypeSet(self.detector.hasMissing(), dims, (subtype,))

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
        
        d = self.getSubDetector()
        for subseq in seq.ravel():
            if not (subseq is Missing):
                d.processSeq(subseq)
        return True
registerTypeScanner(SetScanner)


class StringScanner(TypeScanner):
    good_cls = set([str, unicode, MissingType, numpy.string_, numpy.unicode_])
    unicode_cls = set([unicode, numpy.unicode_])

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.max_nchars = 0

    def getType(self):
        if self.unicode_cls.isdisjoint(self.detector.objectclss):
            ntype = rtypes.TypeBytes
        else:
            ntype = rtypes.TypeString
        if self.max_nchars < 32:
            d = dimensions.Dim(self.max_nchars, tuple(), self.detector.hasMissing())
        else:
            d = dimensions.Dim(UNDEFINED, (True,) * len(self.getDimReps(0)), self.detector.hasMissing())

        dims = dimpaths.DimPath(d)
        return ntype(self.detector.hasMissing(), dims)

    def scan(self, seq):
        has_missing = self.detector.hasMissing()
        if not self.detector.objectclss.issubset(self.good_cls):
            return False
        if(self.max_nchars < 32):
            max_nchars = seq.map(len, otype=int, out_empty=-1, has_missing=has_missing).max(has_missing=False)
            self.max_nchars = max(self.max_nchars, max_nchars)
        return True
registerTypeScanner(StringScanner)

class StringRealScanner(StringScanner):
    parentcls=StringScanner
    missing_str = set(["", "NA", "N/A","NaN", "nan", "--", "?", "null"])
    def __init__(self, detector):
        StringScanner.__init__(self, detector)
        self.has_missing = False
    
    def unregister(self, create_parent=False):
        res = super(StringRealScanner,self).unregister(self, create_parent)
        if create_parent:
            res.max_nchars = self.max_nchars
        return res

    def getType(self):
        return rtypes.TypeReal64(self.detector.hasMissing() or self.has_missing, needscast=True)

    def scan(self, seq):
        res = StringScanner.scan(self, seq)
        if res:
            for elem in seq:
                try:
                    float(elem)
                except ValueError:
                    if elem.__class__ in missing_cls or elem in self.missing_str:
                        self.has_missing = True
                    else:
                        return False
        return res
#registerTypeScanner(StringRealScanner)

class StringIntScanner(StringRealScanner):
    parentcls=StringRealScanner
    
    def unregister(self, create_parent=False):
        res = super(StringIntScanner,self).unregister(self, create_parent)
        if create_parent:
            res.has_missing = self.has_missing
        return res

    def getType(self):
        return rtypes.TypeInt64(self.detector.hasMissing() or self.has_missing, needscast=True)

    def scan(self, seq):
        res = StringScanner.scan(self, seq)
        if res:
            for elem in seq:
                try:
                    int(elem)
                except ValueError:
                    if elem.__class__ in missing_cls or elem in self.missing_str:
                        self.has_missing = True
                    else:
                        return False
        return res
#registerTypeScanner(StringIntScanner)

class SliceScanner(TypeScanner):
    __doc__ = 'Slice scanner'
    typecls = rtypes.TypeSlice
    good_cls = set((slice, MissingType))

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)

    def getType(self):
        if not self.detector.hasMissing():
            pass
        has_missing = self.detector.hasMissing()
        return self.typecls(has_missing)

    def scan(self, seq):
        if not self.detector.objectclss.issubset(self.good_cls):
            return False
        return True
registerTypeScanner(SliceScanner)

class NumberScanner(TypeScanner):
    __doc__ = 'Number scanner, accepts all number objects.'
    typecls = rtypes.TypeNumber
    good_cls = set((bool, float, complex, long, int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float32, numpy.float64, numpy.complex64, numpy.complex128, numpy.bool_, MissingType))

    def __init__(self, detector):
        TypeScanner.__init__(self, detector)
        self.has_nan = False

    def getType(self):
        if not self.detector.hasMissing():
            pass
        has_missing = self.detector.hasMissing() or self.has_nan
        return self.typecls(has_missing)

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
    good_cls = set((float, bool, numpy.float32, numpy.float64, MissingType, long, int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.bool_))
registerTypeScanner(RealScanner)


class IntegerScanner(TypeScanner):
    parentcls = RealScanner
    ustepvals = [256, 65536, 4294967296L, 18446744073709551616L]
    uinttypes = [rtypes.TypeUInt8, rtypes.TypeUInt16, rtypes.TypeUInt32, rtypes.TypeUInt64]
    istepvals = [128, 32768, 2147483648L, 9223372036854775808L]
    inttypes = [rtypes.TypeInt8, rtypes.TypeInt16, rtypes.TypeInt32, rtypes.TypeInt64]
    
    good_cls = set((bool, long, int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.bool_, MissingType))
    
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
        return out_type(self.detector.hasMissing())

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
            minmax = (0,0)
            for cls in self.detector.objectclss:
                if cls is MissingType:
                    continue
                tminmax = self.numpy_minmax[numpy.dtype(int)]
                minmax = (min(tminmax[0],minmax[0]),max(tminmax[1],minmax[1]))

        self.min_val = min(minmax[0], self.min_val)
        self.max_val = max(minmax[1], self.max_val)
        return True
registerTypeScanner(IntegerScanner)

class BoolScanner(TypeScanner):
    parentcls = IntegerScanner
    typecls = rtypes.TypeBool
    good_cls = set((numpy.bool_, bool, MissingType))
registerTypeScanner(BoolScanner)
