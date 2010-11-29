import operator
from collections import defaultdict, Iterable

import rtypes

in_type_casts = defaultdict(list)
cast_exec = defaultdict(dict)

class CheckEnv(object):
    """Object holding attributes to check cast"""
    __slots__ = ['out_type_cls', 'checktypesfunc', 'simtypefunc']
    def __init__(self, out_type_cls, checktypesfunc, simtypefunc):
        self.out_type_cls = out_type_cls
        self.checktypesfunc = checktypesfunc
        self.simtypefunc = simtypefunc

def addCasts(in_type_cls, out_type_cls, checktypesfunc, simtypefunc):
    """Add cast
        in_type_cls: in type class, can also be sequence
        out_type_cls: out type class, can also be sequence
        checktypefunc: function to check actual types for compatibility
        simtypefunc: function to create best matching out_type if possible
    """
    if(isinstance(out_type_cls, Iterable)):
        if(not isinstance(out_type_cls, set)):
            out_type_cls = set(out_type_cls)
    else:
        out_type_cls = set([out_type_cls])
    
    checkenv = CheckEnv(out_type_cls, checktypesfunc, simtypefunc) 
    
    if(isinstance(in_type_cls, Iterable)):
        for incls in in_type_cls:
            in_type_casts[incls].append(checkenv)
    else:
        in_type_casts[in_type_cls].append(checkenv)

def addCastExecFuncs(checktypesfunc, **execfuncs):
    """Add execution functions for casts checked with checktypesfunc
       and for which an execution environment is already available.

       checktypefunc: function to check actual types for compatibility
       execfuncs: execution functions (e.g. ibi for python env., etc.)
    """
    
    cast_exec[checktypesfunc].update(execfuncs)


def findImplicitCastTypes(in_type_cls):
    """Returns type classes that can be casted to without loss of information"""
    res = set(in_type_cls.__mro__[1:])
    res.add(in_type_cls)
    return res 

def canCast(intype, outtype):
    if(not intype.__class__ in in_type_casts):
        return False
    
    pos_casts = in_type_casts[intype.__class__]

    if(isinstance(outtype, rtypes.TypeUnknown)):
        for pos_cast in pos_casts:
            if(outtype.__class__ in pos_cast.out_type_cls and
               pos_cast.checktypesfunc(intype, outtype)):
                return cast_exec[pos_cast.checktypesfunc]
    else:
        for pos_cast in pos_casts:        
            if(outtype.__class__ in pos_cast.out_type_cls):
                otype = pos_cast.simtypefunc(intype, outtype)
                return (otype, cast_exec[pos_cast.checktypesfunc])
    return False

def castImplicitCommonType(type1, type2):
    in1_impli_cls = findImplicitCastTypes(type1.__class__)
    in2_impli_cls = findImplicitCastTypes(type2.__class__)
    out_impli_cls = in1_impli_cls & in2_impli_cls

    while out_impli_cls:
        out_cls = rtypes.mostSpecializedTypesCls(out_impli_cls)
        assert len(out_cls) == 1, \
            "Multiple implicit common types found"
        out_cls = out_cls[0]
        res = out_cls.commonType(type1, type2)
        if(not res is False):
            break
        out_impli_cls.discard(out_cls)
    return res


############
# CAST: check type functions
############
def checkDefault(intype, outtype):#{{{
    if(intype.has_missing and not outtype.has_missing):
        return False
    return True

def simDefault(intype, outtypecls):
    return outtypecls(intype.has_missing)


addCasts(rtypes.TypeNumbers, rtypes.TypeNumbers, checkDefault, simDefault)
