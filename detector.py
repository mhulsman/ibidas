"""Type detector. 

When importing data, types can be represented in multiple ways. 
An integer can be represented as a string, etc. These have 
to be detected (and converted to a standard format). In this module, 
there is a general Detector class, which handles Scanner classes, which
do the actual type checking. 

Also, there is a DimensionEqualizer class, to help the type detector find 
common dimensions between tuple fields. 

"""


import rtypes
from type_attribute_scanner import scanner_protocol

import dimensions
from rtypes import Missing, unknown
import numpy
import array
from collections import defaultdict
from thirdparty.topsort import topsort_levels

__scanchildren__ = defaultdict(list)
def addTypeScanner(newscancls):
    """Adds a new type scanner `newscancls`"""
    __scanchildren__[newscancls.parentcls].append(newscancls)

class TypeDetectError(Exception):
    """Exception for when a matching type cannot be found"""
    def __init__(self, message):
        Exception.__init__(self)
        self.base_message = message
        self.context = []

    def addContext(self, ctmessage):
        """Add context (enclosing scope) to an error"""
        self.context.append(ctmessage)

    def __str__(self):
        return "\n".join(self.context[::-1]) + "\n--> " + self.base_message

class Detector(object):#{{{
    """Used for detection types of objects."""

    def __init__(self, orig_type=unknown, level=100, 
                                    dimension_equalizer=None):
        """Initialize detector object.

        The detector is used to check or detect the type of a 
        (sequence of) objects.
        Parameters
        ----------
        orig_type : current type of the data object(s), optional. If not given
                    or unknown, type will be detected. Otherwise, type will be 
                    checked. 
        level : max depth of nesting within type, optional, default 100
        dimension_equalizer : internal use only. Used to determine equality 
                between dimensions of sequences which are attributes of tuples.

        Algorithm
        ---------
        Generalize: means that we go up in the scanner hiearachy

        (__init__)
        1. Determine acceptable types based on given type
        (processX)
        2. Determine starting scanners, based on data given 
           in first process call.
        3. Generalizes scanners that cannot result in acceptable types
        4. Check data, determine if scanners are compatible. 
           If not, scanner is generalized if its parent has no other 
           children in the list. Afterwards, generalize scanners
           that cannot result in acceptable types.
        (processX)
        5. Go to 4.
        (getType)
        6. Get type.
           - return given type if no scanner is known (i.e. no data has been given)
           - if multiple types, remove double types and types which are more general
             than types that are also in the list
           - if this does not help, perform topological sort on scanners, use types 
             linked to scanners in first level of the sort. 

        """
        assert (level >= 1), "Level should be 1 or higher"
        
        self._orig_type = orig_type
        self._level = level
        
        #classes of processed obj values
        self._value_classes = set()
        
        #active set of representors
        self._scanners = None
        
        #detector-specific attributes from scanners 
        #are stored and shared through this dictionary
        self._scan_attr = {} 
      
        #dimension equalizer, used for detecting equality between dimensions
        #of tuple attributes
        self._dimension_equalizer = dimension_equalizer
       
        #get all acceptable types
        if(self.typeUnknown()):
            self._pos_types = set(orig_type.__class__.getDescendantTypes())
        else:
            self._pos_types = set([orig_type.__class__])


    def typeUnknown(self):
        """Returns if type given at detector initialization is 
        of the unknown type."""
        return self._orig_type.__class__ is rtypes.TypeUnknown

    def _updateValueClasses(self, seq):
        """Updates value_classes parameter, which contains the processed object
        classes and returns (None,Missing)-filtered seq"""
        #update object classes
        #if array is numpy or array.array we can do it more efficient
        if(isinstance(seq, numpy.ndarray) and not seq.dtype == object):
            self._value_classes.add(seq.dtype.type)
        elif(isinstance(seq, array.array)):
            self._value_classes.add(seq[0].__class__)
        else:
            nclasses = set([value.__class__ for value in seq])
            self._value_classes = self._value_classes | nclasses

            #check if we have a none, but the type does not support it
            if(None.__class__ in nclasses or rtypes.MissingType in nclasses):
                #pylint: disable-msg=E1103
                if(not self.typeUnknown() and not self._orig_type.has_missing):
                    raise TypeDetectError, \
                      "Type does not allow missing value, but found one in data"
                #Filter Missing values
                return [elem for elem in seq 
                            if not elem is None or elem is Missing]
        return seq

            
                    
    def hasMissing(self):
        """Returns if the inspected objects did have a None or Missing value,
        or that the given base type allowed for such values."""
        vclass = self._value_classes
        #pylint: disable-msg=E1103
        has_missing = (rtypes.MissingType in vclass or 
                           None.__class__ in vclass or
                               (not self.typeUnknown() 
                      and self._orig_type.has_missing))
        
        return has_missing 

    def getType(self):
        """Returns the type determined by the detector.
        
        Raises
        ------
        TypeDetectError if type cannot be decided
        """
        #get types
        if(not self._scanners):
            return self._orig_type
        restypes = [scanner.getType(self) for scanner in self._scanners]
        #filter None types
        restypes = [t for t in restypes if not t is None]
 
        #if no scanner/type known, return original type
        if(len(restypes) == 0):
            assert (len(self._scanners) == 0), \
                "BUG? no type could be created within constraints"
            return self._orig_type
        
        #check if we only have a single type, if so return it
        if(len(restypes) == 1):
            return restypes[0]
        #filter double types (based on similar classes)
        typecls = set()
        restypes = [t for t in restypes if t.__class__ not in typecls 
                                    and not typecls.add(t.__class__)]
        #check if some types are covered by more specific types
        restypes = rtypes.mostSpecializedTypes(restypes)
        #check again if we only have a single type left
        if(len(restypes) == 1):
            return restypes[0]

        #perform topological sort on scanners
        scantypedict = dict([(t.scanner, t) for t in restypes])
        rscanners = topsort_level_scanners(scantypedict.keys())
        restypes = [scantypedict[r] for r in rscanners]

        if(len(restypes) != 1):
            raise TypeDetectError, \
                (("Cannot decide on types %s, " % str(restypes)) + 
                  "no preferent order has been given")
        
        return restypes[0]

    def _findBetterScanners(self, seq, scans):
        """Find deepest child-scanners of `scanners` that 
           still fit the data

           Parameters
           ----------
           seq : list of data objects
           scanners: list of scanners
        """
        #use limited sample to find better scanner
        if(len(seq) > 100):
            seq = seq[1:100]

        #find better child-scanners 
        res_scanners = sum([self._findBetterScannersHelper(seq, scanner) 
                    for scanner in scans if scanner.scan(self, seq)],[])

        #generalize scanners such that we only have scanners which can result in
        #acceptable types
        res_scanners = self._generalizeScanners([], res_scanners)

        return res_scanners

        
    def _findBetterScannersHelper(self, seq, scanner):
        """Find deepest child-scanners of `scanner` that still fits the data.
           If none found, returns list with `scanner`.

           Parameters
           ----------
           seq : list
           scanner : single scanner
        """
        children_scanners = [child_scan_cls() for child_scan_cls in 
                                __scanchildren__[scanner.__class__]]
        
        ret_scanners = [self._findBetterScannersHelper(seq, child_scanner) 
                    for child_scanner in children_scanners 
                    if child_scanner.scan(self, seq)]
        #if no children scanners, return original scanner
        if(len(ret_scanners) == 0):
            return [scanner]
        return sum(ret_scanners, [])


    def _generalizeScanners(self, remove_scanners, scans):
        """Remove scanners in `remove_scanners`, find parent scanners that 
           still can result in an acceptable type.
           
           Parameters
           ----------
           remove_scanners : list of scanners that are also in `scanners`
           scans : list of scanners

           Raises
           -----
           TypeDetectError : if scanner cannot be generalized
        """
        scans = list(scans)
        for remove_scanner in remove_scanners:
            del scans[scans.index(remove_scanner)]
            parentcls = remove_scanner.parentcls
            #if removed scanner does not occur in ancestor classes 
            #of the `scans`
            if(not any(parentcls in scanner.ancestor_classes 
                                        for scanner in scans)):
                #add it to scanner list (getParent can fail with 
                #TypeDetectError if not suitable parents are known)
                scans.append(remove_scanner.getParent())

        #are types thar can be generated with these scanners allowed
        #by pos_types? otherwise generalize them
        remove_scanners = [scanner for scanner in scans \
                        if not self._pos_types & scanner.out_types]
        if(remove_scanners):
            scans = self._generalizeScanners(remove_scanners, scans)
        return scans
    
    
    def processObj(self, obj):
        """Process an object, update type information if needed.

        Raises
        ------
        TypeError: if data does not fit the given type constraints
        """
        return self.processSeq([obj])

    def processSeq(self, seq):
        """Process an sequence of objects, update type information if needed.

        Raises
        ------
        TypeDetectError: if data does not fit the given type constraints
        """
        #update object classes set
        seq = self._updateValueClasses(seq)
        if(len(seq) == 0):
            return 

        try:
            #check if scanners are available
            if(not self._scanners):
                #otherwise get one from original type
                orig_scanner = scanner_protocol.getScanner(self._orig_type)
                if(orig_scanner):
                    self._scanners = [orig_scanner]
                else:
                    self._scanners = self._findBetterScanners(seq, 
                                                [scanners.AnyScanner()])
                
            if(self.typeUnknown()):
                #determine if there are scanners that do not match the data

                remove_scanners = [scanner for scanner in self._scanners 
                                if not scanner.collect(self, seq)]
                while(remove_scanners):
                    #if scanners fail, generalize them (go to parent scanners)
                    nscanners = self._generalizeScanners(remove_scanners, 
                                                                self._scanners)
                    rescan = set(nscanners) - set(self._scanners)
                    self._scanners = nscanners
                    remove_scanners = [scanner for scanner in rescan 
                                    if not scanner.collect(self, seq)]
            else:
                #check if data adheres to a certain type.
                #the try/except structure is a bit complex to make certain 
                #that the correct error message is given when checking fails
                remove_scanners = []
                for scanner in self._scanners:
                    try:
                        if not scanner.check(self, seq):
                            raise TypeDetectError, "Data check failed"
                    except TypeDetectError, last_error:
                        remove_scanners.append(scanner)
                if(remove_scanners):
                    try:
                        self._scanners = self._generalizeScanners(
                                              remove_scanners, self._scanners)
                    except TypeDetectError, error:
                        raise last_error

        except TypeDetectError, error:
            #on type detect error, add context
            #to help the user if error occurs some
            error.addContext("- matching failed within type constraint: %s" % 
                                                             self._orig_type)
            raise error

    #}}}

class DimensionEqualizer(object):#{{{
    """Class which functions as connector between different array scanners,
    and determines if the dimensions of the arrays of both scanners are equal.
    If so, it helps to equalize the used dimension ids in the 
    resulting types."""

    def __init__(self):
        self.subarray_eq = None

        self.shapes = {}
        self.equals = {}
        self.last_error = None

        self.cached_key_dimid = None

    def getSubArrayDimensionEqualizer(self):
        """Returns a DimensionEqualizer for a subarray"""
        if(self.subarray_eq is None):
            self.subarray_eq = DimensionEqualizer()
        return self.subarray_eq
    
    def shouldAddShapes(self, detector, dimid):
        """Function called by scanner to deterimine if it should add
        shapes of the arrays it is processing.

        Parameters
        ----------
        detector: detector of the scanner
        dimid:    id of the dimension if given. 
                  - strings and numbers > 0 are considered fixed dimensions, 
                    and are checked
                  - numbers <= 0 are just used as identification
        """
                  
        key = (detector, dimid)
        if(not key in self.equals):
            return True

        #are there similar dimensions?
        if(not self.equals[(detector, dimid)]):
            return False

        return True

    def isFixedDim(self, dimid):
        """Is dimid a fixed dimid?"""
        return isinstance(dimid, str)

    def addShapes(self, shapes, detector, dimid):
        """Add shapes to dimension equalizer for
        (detector, dimid) combination. See
        for parameters shouldAddShape function. """

        self.cached_key_dimid = None

        key = (detector, dimid)
        if(not key in self.shapes):
            equals = set()
            for okey, oshapes in self.shapes.iteritems():
                if(oshapes[0] == shapes):
                    equals.add(okey)
                    self.equals[okey].add(key)

            self.shapes[key] = [shapes]
            self.equals[key] = equals
        else:
            pos = len(self.shapes[key])
            for okey in list(self.equals[key]):
                oshapes = self.shapes[okey]
                if(len(oshapes) <= pos):
                    continue
                if(shapes == self.shapes[okey][pos]):
                    continue
                
                self.equals[key].discard(okey)
                self.equals[okey].discard(key)
                if(okey[1] == dimid and self.isFixedDim(dimid)):
                    self.last_error = TypeDetectError("Dimension equality " + 
                                    "constraint violated. Data unequal to " + 
                                                        str(okey[0]._orig_type))
                    #pylint: disable-msg=E0702
                    raise self.last_error
            self.shapes[key].append(shapes)

    def clearShapes(self):
        for key in self.shapes.keys():
            self.shapes[key] = []
        if(self.subarray_eq):
            self.subarray_eq.clearShapes()

    def calcDims(self, detector):
        """Calculate a dictionary of 
        dimid (fixed and non-fixed) --> real dimid (only fixed)
        for dimid in detector
        """
        if(not self.cached_key_dimid):
            key_dimid = {}
            #first process all fixed dims
            for key, equalkeys in self.equals.iteritems():
                if(not self.isFixedDim(key[1])):
                    continue
                key_dimid[key] = key[1]
                for ekey in equalkeys:
                    if(self.isFixedDim(ekey[1])):
                        #hmm, cannot decide between two given dims...
                        #error? for now we just choose one
                        continue
                    key_dimid[ekey] = key[1]

            #then process all other dims
            for key, equalkeys in self.equals.iteritems():
                if(not key in key_dimid):
                    ndimid = dimensions.getNewDimid()
                    key_dimid[key] = ndimid
                    for ekey in equalkeys:
                        key_dimid[ekey] = ndimid
            
            self.cached_key_dimid = key_dimid

        key_dimid = self.cached_key_dimid
        return dict([(k[1], v) for k, v in key_dimid.iteritems() \
                        if k[0] == detector])
        

#}}}

def topsort_level_scanners(scanner_list):#{{{
    """Performs a topological sort on scanners (based on `after` and 
    `before` fields), and returns first level.
    """
    #create dictionary or scanner-class --> scanner
    scanclsdict = dict([(scan.__class__, scan) for scan in scanner_list])

    #list of scanner classes
    clslist = scanclsdict.keys()

    #create dependency edges
    scandeplist = []
    for scanner in scanner_list:
        for bscancls in scanner.before:
            scandeplist.extend([(scanner.__class__, cls) for cls in 
                            clslist if issubclass(cls, bscancls)])
        for ascancls in scanner.after:
            scandeplist.extend([(cls, scanner.__class__) for cls in 
                            clslist if issubclass(cls, ascancls)])

    #determine if there are classes without dependency 
    #these can be used as first level, instead of using full topological
    #sort
    firstcls = set(clslist) - set(sum(scandeplist, ()))

    if(not firstcls):
        #get first level of toplogical sort
        firstcls = iter(topsort_levels(scandeplist)).next()

    #translate classes back to actual scanners
    return [scanclsdict[cls] for cls in firstcls]#}}}


import scanners
