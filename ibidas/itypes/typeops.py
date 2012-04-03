from collections import defaultdict, Iterable
import operator
import numpy

import rtypes

_delay_import_(globals(),"dimensions")
_delay_import_(globals(),"casts")
_delay_import_(globals(),"..utils","util","cutils","sparse_arrays")
_delay_import_(globals(),"..utils.missing","Missing")


in1_type_ops = defaultdict(set)
in2_type_ops = defaultdict(set)
out_type_ops = defaultdict(set)
op_type_ops = defaultdict(set)

class Operation(object):
    __slots__ = ["check_func", "exec_funcs"]
    def __init__(self, check_func, **exec_funcs):
        self.check_func = check_func
        self.exec_funcs = exec_funcs

def addOps(in1_type_cls, in2_type_cls, ops, operation):
    """Add ops
        in1_type_cls: in type class, can also be sequence
        in2_type_cls: in type class, can also be sequence
        ops: python string representation of operation (e.g. __add__)
             can also be sequence
        operation: an Operation object
    """
    if(isinstance(in1_type_cls, Iterable)):
        for in1cls in in1_type_cls:
            in1_type_ops[in1cls].add(operation)
    else:
        in1_type_ops[in1_type_cls].add(operation)
    
    if(isinstance(in2_type_cls, Iterable)):
        for in2cls in in2_type_cls:
            in2_type_ops[in2cls].add(operation)
    else:
        in2_type_ops[in2_type_cls].add(operation)

    if(isinstance(ops, Iterable)
                        and not isinstance(ops, str)):
        for op in ops:
            op_type_ops[op].add(operation)
    else:
        op_type_ops[ops].add(operation)


def unop_type(intype, op):
    operations = op_type_ops[op] & in1_type_ops[intype.__class__]

    for oper in operations:
        res = oper.check_func(intype, op)
        if(not res is False):
            return (res, oper) 
    raise RuntimeError, "Could not find a valid operation handler for " + \
            op + " on " + str(intype) 
        

def binop_type(ltype, rtype, op, outtype=None):
    operations = op_type_ops[op] & in1_type_ops[ltype.__class__] & \
                                    in2_type_ops[rtype.__class__]    
    for oper in operations:
        res = oper.check_func(ltype, rtype, op)
        if(not res is False):
            return (res, oper) 

    raise RuntimeError, "Could not find a valid operation handler for " + \
            str(ltype) + " " + op + " " + str(rtype)
        

def check_arith(in1_type, in2_type, op):#{{{
    in1_impli_cls = casts.findImplicitCastTypes(in1_type.__class__)
    in2_impli_cls = casts.findImplicitCastTypes(in2_type.__class__)
    out_impli_cls = in1_impli_cls & in2_impli_cls
    
    out_impli_cls = [ocls for ocls in out_impli_cls \
                            if issubclass(ocls,rtypes.TypeNumber)]
    if(not out_impli_cls):
        return False
  
    out_impli_cls = rtypes.mostSpecializedTypesCls(out_impli_cls)
    assert len(out_impli_cls) == 1, "Multiple output types for " + \
                  "arithmetic operation found: " + str(out_impli_cls)
    out_cls = out_impli_cls[0]
    
    out_type = out_cls(in1_type.has_missing or in2_type.has_missing)
    return out_type#}}}

numpy_cmp = {'__eq__':numpy.equal,#{{{
            '__ne__':numpy.not_equal,
            '__le__':numpy.less_equal,
            '__ge__':numpy.greater_equal,
            '__lt__':numpy.less,
            '__gt__':numpy.greater}

numpy_arith = { '__add__':numpy.add,
                '__radd__':numpy.add,
                '__sub__':numpy.subtract,
                '__rsub__':numpy.subtract,
                '__mul__':numpy.multiply,
                '__rmul__':numpy.multiply,
                '__mod__':numpy.mod,
                '__rmod__':numpy.mod,
                '__div__':numpy.divide,
                '__rdiv__':numpy.divide,
                '__and__':numpy.bitwise_and,
                '__rand__':numpy.bitwise_and,
                '__or__':numpy.bitwise_or,
                '__ror__':numpy.bitwise_or,
                '__xor__':numpy.bitwise_xor,
                '__rxor__':numpy.bitwise_xor,
                }

numpy_unary_arith = {
    "__invert__":numpy.invert,
    "__neg__":numpy.negative,
    "__abs__":numpy.abs
    }

reverse_op = {'__eq__':'__eq__',
            '__ne__':'__ne__',
            '__le__':'__ge__',
            '__ge__':'__le__',
            '__lt__':'__gt__',
            '__gt__':'__lt__',
            '__add__':'__radd__',
            '__radd__':'__add__',
            '__sub__':'__rsub__',
            '__rsub__':'__sub__',
            '__mul__':'__rmul__',
            '__rmul__':'__mul__',
            '__mod__':'__rmod__',
            '__rmod__':'__mod__',
            '__div__':'__rdiv__',
            '__rdiv__':'__div__',
            '__and__':'__rand__',
            '__rand__':'__and__',
            '__or__':'__ror__',
            '__ror__':'__or__',
            '__xor__':'__rxor__',
            '__rxor__':'__xor__',
            }#}}}

def exec_arith(data, type1, type2, typeo, op):
    data1,data2 = data
    if(data1 is Missing or data2 is Missing):
        return Missing
    if util.numpy16up:
        return numpy_arith[op](data1, data2, dtype=typeo.toNumpy())
    else:
        return numpy_arith[op](data1, data2, sig=typeo.toNumpy())

arith_ops = set(numpy_arith.keys())
addOps(rtypes.TypeNumbers, rtypes.TypeNumbers, arith_ops, Operation(check_arith, py=exec_arith))


def check_cmp(in1_type, in2_type, op):#{{{
    out_cls = rtypes.TypeBool
    out_type = out_cls(in1_type.has_missing or in2_type.has_missing)
    return out_type#}}}

def exec_cmp(data, type1, type2, typeo, op):
    #a numpy bug gives all true arrays when using
    #bool as outtype in comparison
    assert isinstance(typeo,rtypes.TypeBool),"Comparison should have result type bool"
    return numpy_cmp[op](data[0], data[1])


cmp_ops = set(numpy_cmp.keys())
addOps(rtypes.TypeNumbers, rtypes.TypeNumbers, cmp_ops, Operation(check_cmp, py=exec_cmp))

#CHECKED UP TO HERE


def exec_cmpgeneral(data1, data2, type1, type2, typeo, op):
    assert isinstance(typeo,rtypes.TypeBool),"Comparison should have result type bool"
    
    #a numpy bug gives all true arrays when using
    #bool as outtype in comparison
    res = getattr(data1, op)(data2)
    if(res is NotImplemented):
        res = getattr(data2, reverse_op[op])(data1)
    if(res is NotImplemented):
        raise RuntimeError, "Not implemented error in exec_cmpgeneral"
    return res

addOps(rtypes.TypeStrings, rtypes.TypeStrings, cmp_ops, Operation(check_cmp, py=exec_cmpgeneral))

def check_cmpset(in1_type, in2_type, op):#{{{
    
    if(isinstance(in1_type, rtypes.TypeSet) and 
                    isinstance(in2_type, rtypes.TypeSet)):
        pass
    elif isinstance(in1_type, rtypes.TypeSet) and op in set_funcs:
        pass
    elif isinstance(in2_type, rtypes.TypeSet) and op in set_funcs:
        pass
    else:    
        return False
    return check_cmp(in1_type, in2_type, op)#}}}
    

set_funcs = {'__or__':"union", "__ror__":"union", "__and__":"intersection", 
        "__rand__":"intersection",  "__sub__":"difference", 
        "__rsub__":"difference", "__xor__":"symmetric_difference",
        "__rxor__":"symmetric_difference",
        "__le__":"issubset", "__ge__":"issuperset"}


def exec_object_func(data1, data2, type1, type2, otype, op):
    funcget = operator.attrgetter(op)
    if(isinstance(type1, rtypes.TypeAny) and type1.has_missing or
       isinstance(type2, rtypes.TypeAny) and type2.has_missing):
        def func(x, y):
            if(x is Missing or y is Missing):
                return Missing
            else:
                return funcget(x)(y)
    else:  
        func = lambda x, y: funcget(x)(y)
    if(isinstance(data1, numpy.ndarray) and 
                                  not type1.__class__ is rtypes.TypeArray):
        if(isinstance(data2, numpy.ndarray) and 
                                      not type2.__class__ is rtypes.TypeArray):
            res = util.darray([func(left, right) 
                                 for left, right in zip(data1, data2)], 
                                 otype.toNumpy()).view(sparse_arrays.FullSparse)
        else:
            res = util.darray([func(left, data2) 
                                 for left in data1], 
                                 otype.toNumpy()).view(sparse_arrays.FullSparse)
    else:
        if(isinstance(data2, numpy.ndarray) and 
                                      not type2.__class__ is rtypes.TypeArray):
            res = util.darray([func(data1, right) 
                                 for right in data2], 
                                 otype.toNumpy()).view(sparse_arrays.FullSparse)
        else:
            res = func(data1, data2)
            
            if(otype):
                res = otype.toNumpy().type(res)
    return res   

def exec_arrayarray(data1, data2, type1, type2, typeo, op):
    #a numpy bug gives all true arrays when using
    #bool as outtype in comparison
    if(op == "__add__"):
        return numpy.concatenate((data1, data2)).view(sparse_arrays.FullSparse)
    elif(op == "__radd__"):
        return numpy.concatenate((data2, data1)).view(sparse_arrays.FullSparse)
    else:
        raise RuntimeError, "Unrecognized operation " + op

def check_arrayarray(in1_type, in2_type, op):#{{{
    
    if(not (isinstance(in1_type, rtypes.TypeArray) and 
                    isinstance(in2_type, rtypes.TypeArray))):
        return False

    in1_subtype = in1_type.subtypes[0]
    in2_subtype = in2_type.subtypes[0]
    subtype = casts.castImplicitCommonType(in1_subtype, in2_subtype)
    
    odim1 = in1_type.dims[0]
    odim2 = in2_type.dims[0]

    #FIX variable
    ndim = dimensions.Dim(UNDEFINED, 
                          odim1.variable or odim2.variable,  
                          odim1.has_missing or odim2.has_missing)
    return rtypes.TypeArray(in1_type.has_missing or in2_type.has_missing, 
                          (ndim,), (subtype,))
    #}}}
array_ops = set(["__add__", "__radd__"])
addOps(rtypes.TypeArrays, rtypes.TypeArrays, 
       array_ops, Operation(check_arrayarray, py=exec_arrayarray))



def exec_cmpset(data1, data2, type1, type2, typeo, op):
    #a numpy bug gives all true arrays when using
    #bool as outtype in comparison
    rop = reverse_op[op]
    if(op in set_funcs):
        func = set_funcs[op]
        rfunc = set_funcs[rop]
    else:
        func = op
        rfunc = rop
    if(isinstance(type1, rtypes.TypeSet)):
        return exec_object_func(data1, data2, type1, type2, typeo, func)
    else:
        return exec_object_func(data2, data1, type2, type1, typeo, rfunc)



def check_setset(in1_type, in2_type, op, out_type):#{{{
    
    if(isinstance(in1_type, rtypes.TypeSet) and 
                    isinstance(in2_type, rtypes.TypeSet)):
        pass
    elif(isinstance(in1_type, rtypes.TypeSet) and 
                                    (op in set_funcs and op != "__rsub__")):
        pass
    elif(isinstance(in2_type, rtypes.TypeSet) and 
                                    (op in set_funcs and op != "__sub__")):
        pass
    else:    
        return False

    in1_subtype = in1_type.subtypes[0]
    in2_subtype = in2_type.subtypes[0]
    subtype = casts.castImplicitCommonType(in1_subtype, in2_subtype)
    
    odim1 = in1_type.dims[0]
    odim2 = in2_type.dims[0]
    ndim = dimensions.Dim(UNDEFINED, 
                          odim1.variable or odim2.variable, 
                          odim1.has_missing or odim2.has_missing)
    return rtypes.TypeSet(in1_type.has_missing or in2_type.has_missing, 
                          (ndim,), (subtype,))
    #}}}

def exec_setset(data1, data2, type1, type2, typeo, op):
    #a numpy bug gives all true arrays when using
    #bool as outtype in comparison
    rop = reverse_op[op]
    func = set_funcs[op]
    rfunc = set_funcs[rop]

    if(isinstance(type1, rtypes.TypeSet)):
        return exec_object_func(data1, data2, type1, type2, typeo, func)
    else:
        return exec_object_func(data2, data1, type2, type1, typeo, func)

setchange_ops = set(["__or__", "__ror__", "__and__", "__rand__", "__sub__", 
                                        "__rsub__", "__xor__", "__rxor__"])


addOps(rtypes.TypeArrays, rtypes.TypeArrays, 
       cmp_ops, Operation(check_cmpset, py=exec_cmpset))
addOps(rtypes.TypeArrays, rtypes.TypeArrays, 
       setchange_ops, Operation(check_setset, py=exec_setset))


def check_any(in1_type, in2_type, op):#{{{
    if(in1_type.__class__ is rtypes.TypeUnknown or 
       in2_type.__class__ is rtypes.TypeUnknown):
        return rtypes.unknown
    else:
        return rtypes.TypeAny(in1_type.has_missing or in2_type.has_missing)
#}}}

any_types = set([rtypes.TypeUnknown, rtypes.TypeAny])
all_ops = set(reverse_op.keys())

addOps(any_types, any_types, all_ops, 
       Operation(check_any, py=exec_object_func))

def check_unary_any(in_type, op):
    return in_type

def exec_object_unaryfunc(data1, type1, otype, func):
    funcget = operator.attrgetter(func)
    if(isinstance(type1, rtypes.TypeAny) and type1.has_missing):
        def func(x):
            if(x is Missing):
                return x
            else:
                return funcget(x)()
    else:  
        func = lambda x: funcget(x)()

    if(isinstance(data1, numpy.ndarray) and 
                                    not type1.__class__ is rtypes.TypeArray):
        func = numpy.vectorize(func)
        res = numpy.cast[otype.toNumpy()](func(data1)).view(sparse_arrays.FullSparse)
    else:
        res = func(data1)
            
        if(otype):
            res = otype.toNumpy().type(res)
    return res   


unary_ops = set(numpy_unary_arith.keys())
#addOps(rtypes.TypeNumbers, (), rtypes.TypeNumbers, unary_ops, 
#       Operation(check_unary_any, py=exec_object_unaryfunc))

def check_unary_arith(in_type, op):#{{{
    in_impli_cls = casts.findImplicitCastTypes(in_type.__class__)
    out_impli_cls = in_impli_cls
    
    out_impli_cls = [ocls for ocls in out_impli_cls \
                            if issubclass(ocls,rtypes.TypeNumber)]
    if(not out_impli_cls):
        return False
  
    out_impli_cls = rtypes.mostSpecializedTypesCls(out_impli_cls)
    assert len(out_impli_cls) == 1, "Multiple output types for " + \
    "arithmetic operation found: " + str(out_impli_cls)
    out_cls = out_impli_cls[0]
    
    return out_cls(in_type.has_missing)#}}}


def exec_unary_arith(data1, intype, outtype, op):
    if util.numpy16up:
        return numpy_unary_arith[op](data1, dtype=outtype.toNumpy())
    else:
        return numpy_unary_arith[op](data1, sig=outtype.toNumpy())

addOps(rtypes.TypeNumbers, (), unary_ops, 
        Operation(check_unary_arith, py=exec_unary_arith))



