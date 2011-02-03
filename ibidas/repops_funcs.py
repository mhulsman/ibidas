import operator
import repops
from constants import *
from itypes import rtypes
_delay_import_(globals(),"utils","util")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"representor")
_delay_import_(globals(),"wrappers","wrapper_py")
_delay_import_(globals(),"itypes", "casts","dimpaths","dimensions")

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

        if(self.type is None):
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
        
        #cannot find anything
        res = []
        for field,slice in kwargs.iteritems():
            if(isinstance(slice, slices.Slice)):
                res.append((str(slice.type) + " ",field))
            else:
                res.append(("", field + "=" + str(slice)))
        res.sort(key=operator.itemgetter(1))
        res = [a + b for a,b in res]
        raise RuntimeError, "Cannot find func " + str(self.__class__.__name__)  + " with signature (" + ", ".join(res) + ")"
        
       

class UnaryFuncOp(repops.UnaryOpRep, Func):
    def __init__(self, source, *params, **kwargs):
        if(not isinstance(source, representor.Representor)):
            source = wrapper_py.rep(source)
        repops.UnaryOpRep.__init__(self,source, *params,**kwargs)

class UnaryFuncElemOp(UnaryFuncOp):
    def _process(self, source, **kwargs):
        if not source._state & RS_TYPES_KNOWN:
            return

        nslices = []
        for pos, slice in enumerate(source._slices):
            sig, nkwargs, outparam = self._findSignature(slice=slice,**kwargs)
            if(len(source._slices) > 0):
                outparam = outparam.withNumber(pos)
            nslices.append(slices.UnaryFuncElemOpSlice(self.__class__.__name__, sig, outparam, **nkwargs))
        return self._initialize(tuple(nslices),source._state)

class UnaryFuncSeqOp(UnaryFuncOp):
    def _process(self, source, dim=None, **kwargs):
        if not source._state & RS_TYPES_KNOWN:
            return

        selpath = dimpaths.identifyUniqueDimPathSource(source,dim)

        nslices = []
        found = False
        for pos, slice in enumerate(source._slices):
            lastposs = slice.dims.matchDimPath(selpath)
            for lastpos in lastposs:
                found = True
                packdepth = len(slice.dims) - lastpos
                sig, nkwargs, outparam = self._findSignature(slice=slice, packdepth=packdepth, **kwargs)
                if(len(source._slices) > 0):
                    outparam = outparam.withNumber(pos)
                slice = slices.UnaryFuncSeqOpSlice(self.__class__.__name__, sig, outparam, **nkwargs)
            nslices.append(slice)
        if(not found):
            raise RuntimeError, "No slice with dims to apply " + self.__class__.__name__
        return self._initialize(tuple(nslices),source._state)

class UnaryFuncAggregateOp(UnaryFuncOp):
    def _process(self, source, dim=None, **kwargs):
        if not source._state & RS_TYPES_KNOWN:
            return

        selpath = dimpaths.identifyUniqueDimPathSource(source,dim)

        nslices = []
        found = False
        for pos, slice in enumerate(source._slices):
            lastposs = slice.dims.matchDimPath(selpath)
            if(lastposs):
                slice = self._prepareSlice(slice)
                found = True
            for lastpos in lastposs:
                packdepth = len(slice.dims) - lastpos
                sig, nkwargs, outparam = self._findSignature(slice=slice, packdepth=packdepth, **kwargs)
                if(len(source._slices) > 1):
                    outparam = outparam.withNumber(pos)
                slice = slices.UnaryFuncAggregateOpSlice(self.__class__.__name__, sig, outparam, **nkwargs)
            nslices.append(slice)
        if(not found):
            raise RuntimeError, "No slice with dims to apply " + self.__class__.__name__
        return self._initialize(tuple(nslices),source._state)

    def _prepareSlice(self,slice):
        return slice

class BinaryFuncOp(repops.MultiOpRep, Func):
    def __init__(self, lsource, rsource, **kwargs):
        if(not isinstance(lsource, representor.Representor)):
            lsource = repops.PlusPrefix(wrapper_py.rep(lsource))
        if(not isinstance(rsource, representor.Representor)):
            rsource = repops.PlusPrefix(wrapper_py.rep(rsource))
        repops.MultiOpRep.__init__(self,(lsource,rsource), **kwargs)

        
class BinaryFuncElemOp(BinaryFuncOp):
    _allow_partial_bc = True
    def _process(self, sources, **kwargs):
        lsource,rsource = sources
        state = lsource._state & rsource._state
        if not state & RS_TYPES_KNOWN:
            return

        if(isinstance(lsource,repops.PlusPrefix) or isinstance(rsource,repops.PlusPrefix)):
            mode = "pos"
        else:
            mode = "dim"

        nslices = []
        nslice = max(len(lsource._slices), len(rsource._slices))
        for pos, binslices in enumerate(util.zip_broadcast(lsource._slices, rsource._slices)):
            binslices = self._prepareSlices(*binslices)
            (lslice,rslice),plans = slices.broadcast(binslices,mode)
            sig, nkwargs, outparam = self._findSignature(left=lslice,right=rslice,**kwargs)
            if(isinstance(outparam, rtypes.TypeUnknown)):
                if(binslices[0].name == binslices[1].name):
                    name = binslices[0].name
                elif(binslices[0].name == "data"):
                    name = binslices[1].name
                elif(binslices[1].name == "data"):
                    name = binslices[0].name
                else:
                    name = "result"
                outparam = Param(name, outparam)
            if(nslice > 1):
                outparam = outparam.withNumber(pos)
            nslices.append(slices.BinFuncElemOpSlice(self.__class__.__name__, sig,\
                         outparam, allow_partial_bc=self._allow_partial_bc, **nkwargs))
        return self._initialize(tuple(nslices),state)

    def _prepareSlices(self, lslice, rslice):
        return (lslice,rslice)


class WithinSignature(FuncSignature):
    _allow_partial_bc = False
    def check(self, left, right):#{{{
        in1_type = left.type
        in2_type = right.type

        out_cls = rtypes.TypeBool
        out_type = out_cls(in1_type.has_missing or in2_type.has_missing)
        return out_type#}}}

within_sig = WithinSignature("within")

class Within(BinaryFuncElemOp):
    _sigs = [within_sig]
    def _prepareSlices(self,lslice,rslice):
        return (lslice, slices.PackArraySlice(rslice,1))

class BinArithSignature(FuncSignature):
    def check(self, left, right):
        in1_type = left.type
        in2_type = right.type
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
bin_arithsig = BinArithSignature("simple_arith")

class SetSetSignature(FuncSignature):
    def check(self, left, right):
        in1_type = left.type
        in2_type = right.type
        if(not isinstance(left.type, rtypes.TypeSet) or not isinstance(right.type, rtypes.TypeSet)):
            return False
        
        subout_type = casts.castImplicitCommonType(left.type.subtypes[0],right.type.subtypes[0])
        
        ldim = left.type.dims[0]
        rdim = right.type.dims[0]
        dim = dimensions.Dim(UNDEFINED,(True,) * len(left.dims),ldim.has_missing or rdim.has_missing, name=ldim.name + "_" + rdim.name)
        dims = dimpaths.DimPath(dim)

        out_type = rtypes.TypeSet(has_missing=left.type.has_missing or right.type.has_missing, subtypes=(subout_type,), dims=dims)
        return out_type#}}}
setset_sig = SetSetSignature("simple_arith")

@repops.delayable()
class Add(BinaryFuncElemOp):
   _sigs = [bin_arithsig]
@repops.delayable()
class Subtract(BinaryFuncElemOp):
   _sigs = [bin_arithsig, setset_sig]
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
   _sigs = [bin_arithsig, setset_sig]
@repops.delayable()
class Or(BinaryFuncElemOp):
   _sigs = [bin_arithsig, setset_sig]
@repops.delayable()
class Xor(BinaryFuncElemOp):
   _sigs = [bin_arithsig, setset_sig]
@repops.delayable()
class Power(BinaryFuncElemOp):
   _sigs = [bin_arithsig]


class CompareSignature(FuncSignature):
    def __init__(self,name,clss):
        self.comparecls = clss
        FuncSignature.__init__(self,name)

    def check(self, left, right):#{{{
        in1_type = left.type
        in2_type = right.type

        if(not isinstance(in1_type, self.comparecls) or not isinstance(in2_type, self.comparecls)):
            return False

        out_cls = rtypes.TypeBool
        out_type = out_cls(in1_type.has_missing or in2_type.has_missing)

        return out_type#}}}
       
compareanysig = CompareSignature("simple_cmp",rtypes.TypeAny)
comparesetsig = CompareSignature("simple_cmp",rtypes.TypeSet)
comparestringsig = CompareSignature("string_cmp",rtypes.TypeString)

@repops.delayable()
class Equal(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]
@repops.delayable()
class NotEqual(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]

@repops.delayable()
class LessEqual(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]
@repops.delayable()
class Less(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]
    
@repops.delayable()
class GreaterEqual(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]
@repops.delayable()
class Greater(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]


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

class UnaryFixShapeSignature(FuncSignature):
    def __init__(self,name,check_dependent=True):
        self.check_dependent = check_dependent
        FuncSignature.__init__(self,name)

    def check(self,slice,packdepth):
        if(self.check_dependent):
            for pos,i in enumerate(range(len(slice.dims) - packdepth + 1, len(slice.dims))):
                d = slice.dims[i]
                if len(d.dependent) > pos and d.dependent[pos] == True:
                    return False
        return True

class UnaryTypeToTypeSignature(UnaryFixShapeSignature):
    def __init__(self, name, itypecls, otypecls=None, check_dependent=True):
        self.itypecls = itypecls
        self.otypecls = otypecls
        UnaryFixShapeSignature.__init__(self, name, check_dependent)

    def check(self, slice, packdepth, **kwargs):#{{{
        if not UnaryFixShapeSignature.check(self,slice,packdepth):
            return False

        in_type = slice.type
        if(not isinstance(in_type, self.itypecls)):
            return False

        if(self.otypecls is None):
            nstype = in_type
        else:
            nstype = self.otypecls(in_type.has_missing)
        return Param(slice.name, nstype)#}}}

any_nodepsig = UnaryTypeToTypeSignature("any_nodep", rtypes.TypeAny, rtypes.TypePlatformInt, check_dependent=False)
sortablesig = UnaryTypeToTypeSignature("sortable", (rtypes.TypeScalar, rtypes.TypeString, rtypes.TypeTuple), rtypes.TypePlatformInt)

boolsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeBool, rtypes.TypeBool)
numbersig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeNumber)
int_tointsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeInteger,rtypes.TypePlatformInt)
float_tofloatsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeInteger,rtypes.TypeReal64)
number_tofloatsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeNumber,rtypes.TypeReal64)
number_tointsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeNumber,rtypes.TypePlatformInt)

@repops.delayable()
class ArgSort(UnaryFuncSeqOp):
    _sigs = [sortablesig]

@repops.delayable()
class Pos(UnaryFuncSeqOp):
    _sigs = [any_nodepsig]


@repops.delayable()
class Any(UnaryFuncAggregateOp):
    _sigs = [boolsig]

@repops.delayable()
class All(UnaryFuncAggregateOp):
    _sigs = [boolsig]

@repops.delayable()
class Max(UnaryFuncAggregateOp):
    _sigs = [numbersig]

@repops.delayable()
class Min(UnaryFuncAggregateOp):
    _sigs = [numbersig]

@repops.delayable()
class Sum(UnaryFuncAggregateOp):
    _sigs = [int_tointsig, float_tofloatsig]

@repops.delayable()
class Mean(UnaryFuncAggregateOp):
    _sigs = [number_tofloatsig]

@repops.delayable()
class ArgMax(UnaryFuncAggregateOp):
    _sigs = [number_tointsig]

@repops.delayable()
class ArgMin(UnaryFuncAggregateOp):
    _sigs = [number_tointsig]

@repops.delayable()
class Median(UnaryFuncAggregateOp):
    _sigs = [number_tofloatsig]


class CountSignature(FuncSignature):
    def check(self, slice, packdepth):#{{{
        if(packdepth > 1):
            return False
        
        nstype = rtypes.TypePlatformInt(len(slice.dims) > 1 and slice.dims[-2].has_missing)
        return Param(slice.name, nstype)#}}}
countsig = CountSignature("count")

@repops.delayable()
class Count(UnaryFuncAggregateOp):
    _sigs = [countsig]

class SetSignature(UnaryFixShapeSignature):
    def check(self, slice, packdepth):#{{{
        if not UnaryFixShapeSignature.check(self,slice,packdepth):
            return False
        
        has_missing = len(slice.dims) > 1 and slice.dims[-2].has_missing
        subtypes = (slice.type,)
        dim = dimensions.Dim(UNDEFINED,(True,) * len(slice.dims), slice.type.has_missing, name="s" + slice.dims[-packdepth].name)
        nstype = rtypes.TypeSet(has_missing,dimpaths.DimPath(dim),subtypes)
        return Param(slice.name, nstype)#}}}
setsig = SetSignature("set")

class Set(UnaryFuncAggregateOp):
    _sigs = [setsig]

    def _prepareSlice(self,slice):
        return slices.ensure_frozen(slice)
            
