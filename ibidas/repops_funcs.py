import repops
from constants import *
from itypes import rtypes
_delay_import_(globals(),"utils","util")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"representor")
_delay_import_(globals(),"wrappers","wrapper_py")
_delay_import_(globals(),"itypes", "casts")

class FuncSignature(object):
    def __init__(self, signame):
        self.name = signame
    def check(self, **kwargs):
        return True

class Param(object):
    __slots__ = ["name", "type","default"]
    def __init__(self,name, type=None, default=NOVAL):
        assert isinstance(name, str), "Param name should be a string"
        assert type is None or isinstance(type,rtypes.TypeUnknown), "Type should be None or a type from module rtypes"

        self.name = name
        self.type = type
        self.default = default

    def __str__(self):
       if(self.type is None):
           res = self.name
       else:
           res = str(self.type) + " " + self.name
       if(not self.default is NOVAL):
           if(isinstance(self.default,representor.Representor)):
               res += "=" + str(self.default())
           else:
               res += "=" + str(self.default)
       return res

    def check(self,val):
        if(val is NOVAL):
            if(self.default is NOVAL):
                return False
            else:
                return self.default

        if(type is None):
            return val
        else:
            if(not isinstance(val, slices.Slice)):
                return False
            if not isinstance(val.type, self.type.__class__):
                return False
        return val

    def withNumber(self, number):
        return self.__class__(self.name + str(number), self.type, self.default)


class ParamFuncSignature(object):
    __slots__ = ['signame', 'inparams', 'outparam']

    def __init__(self, inparams, outparams, signame=None):
        assert all(isinstance(p, Param) for p in inparams), "Inparams should be of param type"
        assert isinstance(outparam, Param), "Outparamsshould be of param type"
        self.inparams = inparams
        self.outparam = outparam
        
        if(signame is None):
            inparamstr = ",".join(map(str,inparams))
            outparamstr = str(outparam)
            signame = inparamstr + "-->" + outparamstr
        FuncSignature.__init__(signame)
            
    def check(self, **kwargs):
        res = {}
        for param in self.inparams:
            pc = param.check(kwargs.get(param.name,None))
            if(pc is False):
                return False
            res[param.name] = pc
        return (res, self.outparams)

class Func(object):
    _sigs = []

    def _findSignature(self, **kwargs):
        for sig in self._sigs:
            res = sig.check(**kwargs)
            if(res is False):
                continue
            if(isinstance(res,tuple)):
                nkwargs, outdescr = res
            else:
                nkwargs = kwargs.copy()
                outdescr = res
            return (sig, nkwargs, outdescr)
        else:
            res = []
            for field,slice in kwargs.iteritems():
                if(isinstance(slice, slices.Slice)):
                    res.append(str(slice.type) + " " + field)
                else:
                    res.append(field + "=" + str(slice))

            raise RuntimeError, "Cannot find func " + str(self.__class__.__name__)  + " with signature (" + ",".join(res) + ")"
        
       

class UnaryFuncOp(repops.UnaryOpRep, Func):
    def __init__(self, source, **kwargs):
        if(not isinstance(source, representor.Representor)):
            source = wrapper_py.rep(source)
        repops.UnaryOpRep.__init__(self,source, **kwargs)

class UnaryFuncElemOp(UnaryFuncOp):
    def process(self, source, **kwargs):
        if not source._state & RS_TYPES_KNOWN:
            return

        nslices = []
        for pos, slice in enumerate(source._slices):
            kwargs["slice"] = slice
            sig, nkwargs, outparam = self._findSignature(**kwargs)
            outparam = outparam.withNumber(pos)
            nslices.append(slices.UnaryFuncElemOpSlice(self.__class__.__name__, sig, outparam, **nkwargs))
        return self.initialize(tuple(nslices),source._state)

class UnaryFuncSeqOp(UnaryFuncOp):
    def process(self, source, **kwargs):
        if not source._state & RS_TYPES_KNOWN:
            return

        nslices = []
        for pos, slice in enumerate(source._slices):
            slice = slices.PackArraySlice(slice,1)
            kwargs["slice"] = slice
            sig, nkwargs, outparam = self._findSignature(**kwargs)
            outparam = outparam.withNumber(pos)
            slice = slices.UnaryFuncElemOpSlice(self.__class__.__name__, sig, outparam, **nkwargs)
            slice = slices.UnpackArraySlice(slice,1)
            nslices.append(slice)
        return self.initialize(tuple(nslices),source._state)

class UnaryFuncAggregateOp(UnaryFuncOp):
    def process(self, source, **kwargs):
        if not source._state & RS_TYPES_KNOWN:
            return

        nslices = []
        for pos, slice in enumerate(source._slices):
            slice = slices.PackArraySlice(slice,1)
            kwargs["slice"] = slice
            sig, nkwargs, outparam = self._findSignature(**kwargs)
            outparam = outparam.withNumber(pos)
            slice = slices.UnaryFuncElemOpSlice(self.__class__.__name__, sig, outparam, **nkwargs)
            nslices.append(slice)
        return self.initialize(tuple(nslices),source._state)


class BinaryFuncOp(repops.MultiOpRep, Func):
    def __init__(self, lsource, rsource, **kwargs):
        if(not isinstance(lsource, representor.Representor)):
            lsource = repops.PlusPrefix(wrapper_py.rep(lsource))
        if(not isinstance(rsource, representor.Representor)):
            rsource = repops.PlusPrefix(wrapper_py.rep(rsource))
        repops.MultiOpRep.__init__(self,(lsource,rsource), **kwargs)

        
class BinaryFuncElemOp(BinaryFuncOp):
    def process(self, sources, **kwargs):
        lsource,rsource = sources
        state = lsource._state & rsource._state
        if not state & RS_TYPES_KNOWN:
            return

        if(isinstance(lsource,repops.PlusPrefix) or isinstance(rsource,repops.PlusPrefix)):
            mode = "pos"
        else:
            mode = "dim"

        nslices = []
        for pos, binslices in enumerate(util.zip_broadcast(lsource._slices, rsource._slices)):
            (kwargs["lslice"],kwargs["rslice"]),plans = slices.broadcast(binslices,mode)
            sig, nkwargs, outparam = self._findSignature(**kwargs)
            if(isinstance(outparam, rtypes.TypeUnknown)):
                if(binslices[0].name == binslices[1].name):
                    name = binslices[0].name
                else:
                    name = "result"
                outparam = Param(name, outparam)

            outparam = outparam.withNumber(pos)
            nslices.append(slices.BinFuncElemOpSlice(self.__class__.__name__, sig, outparam, **nkwargs))
        return self.initialize(tuple(nslices),state)


class BinArithSignature(FuncSignature):
    def check(self, lslice, rslice):
        in1_type = lslice.type
        in2_type = rslice.type
        if(not isinstance(in1_type, rtypes.TypeNumber) or not isinstance(in2_type, rtypes.TypeNumber)):
            return False

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
bin_arithsig = BinArithSignature("number_number")

@repops.delayable()
class Add(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class Subtract(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class Multiply(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class Modulo(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class Divide(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class FloorDivide(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class And(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class Or(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class Xor(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class Power(BinaryFuncElemOp):
   _sigs = [bin_arithsig]


class CompareSignature(FuncSignature):
    def check(self, lslice, rslice):#{{{
        in1_type = lslice.type
        in2_type = rslice.type
        if(not isinstance(in1_type, rtypes.TypeScalar) or not isinstance(in2_type, rtypes.TypeScalar)):
            return False

        out_cls = rtypes.TypeBool
        out_type = out_cls(in1_type.has_missing or in2_type.has_missing)

        return out_type#}}}
       
comparesig = CompareSignature("scalar_scalar")

@repops.delayable()
class Equal(BinaryFuncElemOp):
    _sigs = [comparesig]
@repops.delayable()
class NotEqual(BinaryFuncElemOp):
    _sigs = [comparesig]

@repops.delayable()
class LessEqual(BinaryFuncElemOp):
    _sigs = [comparesig]
@repops.delayable()
class Less(BinaryFuncElemOp):
    _sigs = [comparesig]
    
@repops.delayable()
class GreaterEqual(BinaryFuncElemOp):
    _sigs = [comparesig]
@repops.delayable()
class Greater(BinaryFuncElemOp):
    _sigs = [comparesig]


class UnaryArithSignature(FuncSignature):
    def check(self, slice):#{{{
        in_type = slice.type
        if(not isinstance(in_type, rtypes.TypeNumber)):
            return False

        return Param(slice.name, in_type)#}}}

unary_arithsig = UnaryArithSignature("number")

@repops.delayable()
class Invert(UnaryFuncElemOp):
    _sigs = [unary_arithsig]

@repops.delayable()
class Abs(UnaryFuncElemOp):
    _sigs = [unary_arithsig]

@repops.delayable()
class Negative(UnaryFuncElemOp):
    _sigs = [unary_arithsig]

class UnaryArraySubtypeToIntegerSignature(FuncSignature):
    def __init__(self, name, subtypecls):
        self.subtypecls = subtypecls
        FuncSignature.__init__(self, name)

    def check(self, slice):#{{{
        in_type = slice.type
        if(not isinstance(in_type, rtypes.TypeArray) or len(in_type.dims) != 1):
            return False

        if(not isinstance(in_type.subtypes[0], self.subtypecls)):
            return False
        
        #FIXME: make integer type platform dependent
        nstype = rtypes.TypeInt64(in_type.subtypes[0].has_missing)
        ntype = rtypes.TypeArray(in_type.has_missing, in_type.dims, (nstype,))
        return Param(slice.name, ntype)#}}}

unary_arrayanysig = UnaryArraySubtypeToIntegerSignature("arrayany", rtypes.TypeAny)
unary_arrayscalarsig = UnaryArraySubtypeToIntegerSignature("arrayscalar", (rtypes.TypeScalar,rtypes.TypeString, rtypes.TypeTuple))

@repops.delayable()
class ArgSort(UnaryFuncSeqOp):
    _sigs = [unary_arrayscalarsig]

@repops.delayable()
class Pos(UnaryFuncSeqOp):
    _sigs = [unary_arrayanysig]


class UnaryArrayBoolToBoolSignature(FuncSignature):
    def check(self, slice):#{{{
        in_type = slice.type
        if(not isinstance(in_type, rtypes.TypeArray) or len(in_type.dims) != 1):
            return False

        if(not isinstance(in_type.subtypes[0], rtypes.TypeBool)):
            return False
        
        #FIXME: make integer type platform dependent
        ntype = rtypes.TypeBool(in_type.subtypes[0].has_missing)
        return Param(slice.name, ntype)#}}}

unary_arrayboolsig = UnaryArrayBoolToBoolSignature("arraybool")


@repops.delayable()
class Any(UnaryFuncAggregateOp):
    _sigs = [unary_arrayboolsig]

@repops.delayable()
class All(UnaryFuncAggregateOp):
    _sigs = [unary_arrayboolsig]

