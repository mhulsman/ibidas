import operator
import repops
from constants import *
from itypes import rtypes
import ops
_delay_import_(globals(),"utils","util")
_delay_import_(globals(),"representor")
_delay_import_(globals(),"wrappers","python")
_delay_import_(globals(),"itypes", "casts","dimpaths","dimensions")

class FuncSignature(object):
    def __init__(self, signame):
        self.name = signame
    def check(self, **kwargs):
        return True

class Param(object):
    __slots__ = ["name", "type","default"]
    def __init__(self,name, type=None, default=NOVAL):
        assert isinstance(name, basestring), "Param name should be a string"
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
            if(not isinstance(val, ops.UnaryOp)):
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

    @classmethod
    def _findSignature(cls, **kwargs):
        for sig in cls._sigs:
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
            if(isinstance(slice, ops.UnaryOp)):
                res.append((str(slice.type) + " ",field))
            else:
                res.append(("", field + "=" + str(slice)))
        res.sort(key=operator.itemgetter(1))
        res = [a + b for a,b in res]
        raise RuntimeError, "Cannot find func " + str(cls.__name__)  + " with signature (" + ", ".join(res) + ")"
        
       

class UnaryFuncOp(repops.UnaryOpRep, Func):
    def __init__(self, source, *params, **kwargs):
        if(not isinstance(source, representor.Representor)):
            source = python.Rep(source,name="data")
        repops.UnaryOpRep.__init__(self,source, *params,**kwargs)

class UnaryFuncElemOp(UnaryFuncOp):
    def _process(self, source, **kwargs):
        if not source._typesKnown():
            return
        nslices = []
        for slice in source._slices:
            slice = self._prepareSlice(slice)
            sig, nkwargs, outparam = self._findSignature(slice=slice,**kwargs)
            s = ops.UnaryFuncElemOp(self.__class__.__name__, sig, outparam, **nkwargs)
            s = self._finishSlice(s)
            nslices.append(s)
        return self._initialize(tuple(nslices))

    def _prepareSlice(self,slice):
        return slice

    def _finishSlice(self, slice):
        return slice

class UnaryFuncDimOp(UnaryFuncOp):
    _slicecls = ops.UnaryFuncSeqOp

    def _process(self, source, dim=None, **kwargs):
        if not source._typesKnown():
            return
        nslices = self._apply(source._slices, dim, **kwargs)
        return self._initialize(tuple(nslices))

    @classmethod
    def _apply(cls, xslices, dim,  **kwargs):
        nslices = []
        found = False
        for slice in xslices:
            selpath = dimpaths.identifyUniqueDimPath([slice.dims],dim)
            lastposs = slice.dims.matchDimPath(selpath)
            if(lastposs):
                slice = cls._prepareSlice(slice)
                found = True
            for lastpos in lastposs:
                packdepth = len(slice.dims) - lastpos
                sig, nkwargs, outparam = cls._findSignature(slice=slice, packdepth=packdepth, **kwargs)
                slice = cls._slicecls(cls.__name__, sig, outparam, **nkwargs)
            if(lastposs):
                slice = cls._finishSlice(slice)
            nslices.append(slice)

        if(not found):
            raise RuntimeError, "No slice with dims to apply " + cls.__name__
        
        return nslices

    @classmethod 
    def _prepareSlice(self,slice):
        return slice
    @classmethod
    def _finishSlice(self, slice):
        return slice

class UnaryFuncAggregateOp(UnaryFuncDimOp):
    _slicecls = ops.UnaryFuncAggregateOp

class BinaryFuncOp(repops.MultiOpRep, Func):
    def __init__(self, lsource, rsource, **kwargs):
        if(not isinstance(lsource, representor.Representor)):
            lsource = repops.PlusPrefix(python.Rep(lsource,name="data"))
        if(not isinstance(rsource, representor.Representor)):
            rsource = repops.PlusPrefix(python.Rep(rsource,name="data"))
        repops.MultiOpRep.__init__(self,(lsource,rsource), **kwargs)

        
class BinaryFuncElemOp(BinaryFuncOp):
    _allow_partial_bc = False
    def _process(self, sources, **kwargs):
        lsource,rsource = sources
        if not lsource._typesKnown() or not rsource._typesKnown():
            return

        if(isinstance(lsource,repops.PlusPrefix) or isinstance(rsource,repops.PlusPrefix)):
            mode = "pos"
        else:
            mode = "dim"
        
        nslices = []
        nslice = max(len(lsource._slices), len(rsource._slices))
        for pos, binslices in enumerate(util.zip_broadcast(lsource._slices, rsource._slices)):
            if(nslice <= 1):
                pos = None
            nslices.append(self._apply(binslices, mode, pos, **kwargs))

        return self._initialize(tuple(nslices))

    @classmethod
    def _apply(cls, binslices, mode, pos=None, **kwargs):
        binslices = cls._prepareSlices(*binslices)
        (lslice,rslice),plans = ops.broadcast(binslices,mode, partial=cls._allow_partial_bc)
        sig, nkwargs, outparam = cls._findSignature(left=lslice,right=rslice,**kwargs)
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
            if(not pos is None and name == "result"):
                outparam = outparam.withNumber(pos)
        s = ops.BinFuncElemOp(cls.__name__, sig,outparam, **nkwargs)
        return s

    @classmethod
    def _prepareSlices(cls, lslice, rslice):
        return (lslice,rslice)


class WithinSignature(FuncSignature):
    def check(self, left, right):#{{{
        in1_type = left.type
        in2_type = right.type

        out_cls = rtypes.TypeBool
        out_type = out_cls(in1_type.has_missing or in2_type.has_missing)
        return out_type#}}}

within_sig = WithinSignature("within")

class Within(BinaryFuncElemOp):
    _allow_partial_bc = True
    _sigs = [within_sig]
    @classmethod
    def _prepareSlices(self,lslice,rslice):
        return (lslice, ops.PackArrayOp(ops.ensure_frozen(rslice),1))

class Contains(BinaryFuncElemOp):
    _allow_partial_bc = True
    _sigs = [within_sig]
    @classmethod
    def _prepareSlices(self,lslice,rslice):
        return (ops.PackArrayOp(ops.ensure_frozen(lslice),1), rslice)

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

class StringAddStringSignature(FuncSignature):
    def check(self, left, right):
        in1_type = left.type
        in2_type = right.type
        if(not isinstance(in1_type, rtypes.TypeString) or not isinstance(in2_type, rtypes.TypeString)):
            return False
       
        if(in1_type.dims[0].shape == UNDEFINED or in2_type.dims[0].shape == UNDEFINED):
            nshape =  UNDEFINED
        else:
            nshape = in1_type.dims[0].shape + in2_type.dims[0].shape

        in1_impli_cls = casts.findImplicitCastTypes(in1_type.__class__)
        in2_impli_cls = casts.findImplicitCastTypes(in2_type.__class__)
        out_impli_cls = in1_impli_cls & in2_impli_cls
        out_impli_cls = rtypes.mostSpecializedTypesCls(out_impli_cls)
        assert len(out_impli_cls) == 1, "Multiple output types for " + \
                    "arithmetic operation found: " + str(out_impli_cls)
        out_cls = out_impli_cls[0]
        
        ndim = dimensions.Dim(nshape,dependent=(True,) * len(left.dims), name = in1_type.dims[0].name + "_" + in2_type.dims[0].name)
        out_type = out_cls(in1_type.has_missing or in2_type.has_missing, dims=dimpaths.DimPath(ndim))
        return out_type#}}}
stringaddstring_sig = StringAddStringSignature("string_add_string")

class ArrayAddArraySignature(FuncSignature):
    def check(self, left, right):
        in1_type = left.type
        in2_type = right.type
        if(not in1_type.__class__ is rtypes.TypeArray or not in2_type.__class__ is rtypes.TypeArray):
            return False

        if(in1_type.dims[1:]):
            subtype1 = rtypes.TypeArray(subtypes=in1_type.subtypes, dims=in1_type.dims[1:])
        else:
            subtype1 = in1_type.subtypes[0]
        
        if(in2_type.dims[1:]):
            subtype2 = rtypes.TypeArray(subtypes=in2_type.subtypes, dims=in2_type.dims[1:])
        else:
            subtype2 = in2_type.subtypes[0]

        nsubtype = casts.castImplicitCommonType(subtype1,subtype2)
        if(nsubtype is False):
            return False
            
        if(in1_type.dims[0].shape == UNDEFINED or in2_type.dims[0].shape == UNDEFINED):
            nshape =  UNDEFINED
        else:
            nshape = in1_type.dims[0].shape + in2_type.dims[0].shape
        
        ndim = dimensions.Dim(nshape,dependent=(True,) * len(left.dims), name = in1_type.dims[0].name + "_" + in2_type.dims[0].name)
        out_type = rtypes.TypeArray(in1_type.has_missing or in2_type.has_missing, dims=dimpaths.DimPath(ndim), subtypes=(nsubtype,))
        return out_type#}}}
arrayaddarray_sig = ArrayAddArraySignature("array_add_array")

class Add(BinaryFuncElemOp):
   _sigs = [bin_arithsig, stringaddstring_sig, arrayaddarray_sig]

class Subtract(BinaryFuncElemOp):
   _sigs = [bin_arithsig, setset_sig]

class Multiply(BinaryFuncElemOp):
   _sigs = [bin_arithsig]

class Modulo(BinaryFuncElemOp):
   _sigs = [bin_arithsig]

class Divide(BinaryFuncElemOp):
   _sigs = [bin_arithsig]

class FloorDivide(BinaryFuncElemOp):
   _sigs = [bin_arithsig]

class BoolOutSignature(FuncSignature):
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
       
boolboolsig = BoolOutSignature("boolbool",rtypes.TypeBool)

class And(BinaryFuncElemOp):
   _sigs = [boolboolsig, bin_arithsig, setset_sig]

class Or(BinaryFuncElemOp):
   _sigs = [boolboolsig, bin_arithsig, setset_sig]

class Xor(BinaryFuncElemOp):
   _sigs = [boolboolsig, bin_arithsig, setset_sig]

class Power(BinaryFuncElemOp):
   _sigs = [bin_arithsig]


       
compareanysig = BoolOutSignature("simple_cmp",rtypes.TypeAny)
comparesetsig = BoolOutSignature("simple_cmp",rtypes.TypeSet)
comparestringsig = BoolOutSignature("string_cmp",rtypes.TypeString)

class Equal(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]

class NotEqual(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]

class LessEqual(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]

class Less(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]
    
class GreaterEqual(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]

class Greater(BinaryFuncElemOp):
    _sigs = [comparestringsig,comparesetsig,compareanysig]

class MergeSignature(FuncSignature):
    def check(self, left, right):
        in1_type = left.type
        in2_type = right.type

        res = casts.castImplicitCommonType(in1_type, in2_type)
        if res is False:
            return False
        return res#}}}
mergesig = MergeSignature("merge")

class Merge(BinaryFuncElemOp):
    _sigs = [mergesig]

class EachSignature(FuncSignature):
    def check(self, slice, eachfunc, dtype=rtypes.unknown):#{{{
        if(not isinstance(dtype,rtypes.TypeUnknown)):
            dtype = rtypes.createType(dtype,len(slice.dims)) 
        nkwargs = {'eachfunc': eachfunc, 'slice':slice}
        return (nkwargs, Param(slice.name, dtype))#}}}
eachsig = EachSignature("each")

class Each(UnaryFuncElemOp):
    _sigs = [eachsig]


class UnaryArithSignature(FuncSignature):
    def check(self, slice):#{{{
        in_type = slice.type
        if(not isinstance(in_type, rtypes.TypeNumber)):
            return False

        return Param(slice.name, in_type)#}}}

unary_arithsig = UnaryArithSignature("number")

class UnaryBoolSignature(FuncSignature):
    def check(self, slice):#{{{
        in_type = slice.type
        if(not in_type.has_missing or not isinstance(in_type, rtypes.TypeBool)):
            return False

        return Param(slice.name, in_type)#}}}

unary_boolsig = UnaryBoolSignature("bool")

class Invert(UnaryFuncElemOp):
    _sigs = [unary_boolsig, unary_arithsig]

class Abs(UnaryFuncElemOp):
    _sigs = [unary_arithsig]

class Negative(UnaryFuncElemOp):
    _sigs = [unary_arithsig]


class ReplaceMissingSig(FuncSignature):
    def check(self, slice, def_value=NOVAL):#{{{
        in_type = slice.type
        nin_type = in_type.copy()
        #FIXME: adapt dim in case of array
        nin_type = nin_type.setHasMissing(False)

        return Param(slice.name, nin_type)#}}}

repmissing = ReplaceMissingSig("repmissing")

class ReplaceMissing(UnaryFuncElemOp):
    _sigs =[repmissing]


class IsMissingSig(FuncSignature):
    def check(self, slice, def_value=NOVAL):#{{{
        o_type = rtypes.TypeBool()
        return Param(slice.name, o_type)#}}}
ismissingsig = IsMissingSig("ismissing")

class IsMissing(UnaryFuncElemOp):
    _sigs =[ismissingsig]

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

boolsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeBool, rtypes.TypeBool)
numbersig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeNumber)
int_tointsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeInteger,rtypes.TypePlatformInt)
float_tofloatsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeReal64,rtypes.TypeReal64)
number_tofloatsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeNumber,rtypes.TypeReal64)
number_tointsig = UnaryTypeToTypeSignature("fixdim", rtypes.TypeNumber,rtypes.TypePlatformInt)

class UnarySortableSignature(UnaryFixShapeSignature):
    def __init__(self, name, check_dependent=True):
        UnaryFixShapeSignature.__init__(self, name, check_dependent)

    def check(self, slice, packdepth, **kwargs):#{{{
        if not UnaryFixShapeSignature.check(self,slice,packdepth):
            return False

        in_type = slice.type
        if(in_type.__class__ == rtypes.TypeArray):
            return False

        nstype = rtypes.TypePlatformInt()
        return Param(slice.name, nstype)#}}}

sortablesig = UnarySortableSignature("sortable")
class Argsort(UnaryFuncDimOp):
    _sigs = [sortablesig]

sortablesig = UnarySortableSignature("sortable")
class Rank(UnaryFuncDimOp):
    _sigs = [sortablesig]

class Pos(UnaryFuncDimOp):
    _sigs = [any_nodepsig]

class CumSum(UnaryFuncDimOp):
    _sigs = [int_tointsig, float_tofloatsig]

class Any(UnaryFuncAggregateOp):
    _sigs = [boolsig]

class All(UnaryFuncAggregateOp):
    _sigs = [boolsig]

class Max(UnaryFuncAggregateOp):
    _sigs = [numbersig]

class Min(UnaryFuncAggregateOp):
    _sigs = [numbersig]

class UnaryConcatenateSignature(UnaryFixShapeSignature):
    def check(self, slice, packdepth, **kwargs):#{{{
        if not UnaryFixShapeSignature.check(self,slice,packdepth):
            return False
        in_type = slice.type
        if(not in_type.__class__ is rtypes.TypeArray):
            return False
        if(not in_type.dims):
            return False

        curdim = in_type.dims[0]
        adim = slice.dims[-packdepth]
        
        if(not curdim.shape is UNDEFINED and not adim.shape is UNDEFINED):
            nshape = curdim.shape * adim.shape
        else:
            nshape = UNDEFINED
        nname = "s" + curdim.name

        #note: in_type.dims[0].dependent will be updated by aggregate slice
        ndim = dimensions.Dim(nshape, in_type.dims[0].dependent, has_missing=in_type.dims[0].has_missing, name = nname)
        nin_type = in_type._updateDepDim(0,ndim)

        return Param(slice.name, nin_type)#}}}

concatenate_sig = UnaryConcatenateSignature("arrayarray")

class UnaryStringConcatenateSignature(UnaryFixShapeSignature):
    def check(self, slice, packdepth, **kwargs):#{{{
        if not UnaryFixShapeSignature.check(self,slice,packdepth):
            return False
        in_type = slice.type
        if(not isinstance(in_type,rtypes.TypeString)):
            return False
        if(not in_type.dims):
            return False

        curdim = in_type.dims[0]
        adim = slice.dims[-packdepth]
        
        if(not curdim.shape is UNDEFINED and not adim.shape is UNDEFINED):
            nshape = curdim.shape * adim.shape
        else:
            nshape = UNDEFINED
        nname = "s" + curdim.name

        #note: in_type.dims[0].dependent will be updated by aggregate slice
        ndim = dimensions.Dim(nshape, in_type.dims[0].dependent, has_missing=in_type.dims[0].has_missing, name = nname)
        nin_type = in_type._updateDepDim(0,ndim)

        return Param(slice.name, nin_type)#}}}
strconcatenate_sig = UnaryStringConcatenateSignature("stringstring")

class Sum(UnaryFuncAggregateOp):
    _sigs = [int_tointsig, float_tofloatsig, concatenate_sig, strconcatenate_sig]

class Mean(UnaryFuncAggregateOp):
    _sigs = [number_tofloatsig]

class Argmax(UnaryFuncAggregateOp):
    _sigs = [number_tointsig]

class Argmin(UnaryFuncAggregateOp):
    _sigs = [number_tointsig]

class Median(UnaryFuncAggregateOp):
    _sigs = [number_tofloatsig]

class Std(UnaryFuncAggregateOp):
    _sigs = [number_tofloatsig]

class CountSignature(FuncSignature):
    def check(self, slice, packdepth):#{{{
        if(packdepth > 1):
            return False
        
        nstype = rtypes.TypePlatformInt(len(slice.dims) > 1 and slice.dims[-2].has_missing)
        return Param(slice.name, nstype)#}}}
countsig = CountSignature("count")

class Count(UnaryFuncAggregateOp):
    _sigs = [countsig]

class SetSignature(UnaryFixShapeSignature):
    def check(self, slice, packdepth):#{{{
        if not UnaryFixShapeSignature.check(self,slice,packdepth):
            return False
        
        has_missing = slice.dims[-1].has_missing
        subtypes = (slice.type,)
        dim = dimensions.Dim(UNDEFINED,(True,) * len(slice.dims), slice.type.has_missing, name="s" + slice.dims[-packdepth].name)
        nstype = rtypes.TypeSet(has_missing,dimpaths.DimPath(dim),subtypes)
        return Param(slice.name, nstype)#}}}
setsig = SetSignature("set")

class Set(UnaryFuncAggregateOp):
    _sigs = [setsig]

    @classmethod
    def _prepareSlice(self,slice):
        return ops.ensure_frozen(slice)

class UniqueSignature(UnaryFixShapeSignature):
    def check(self, slice, packdepth):#{{{
        if not UnaryFixShapeSignature.check(self,slice,packdepth):
            return False
        
        has_missing = slice.dims[-1].has_missing
        subtypes = (rtypes.TypePlatformInt(),)
        dim = dimensions.Dim(UNDEFINED,(True,) * len(slice.dims), slice.type.has_missing, name="s" + slice.dims[-packdepth].name)
        nstype = rtypes.TypeArray(has_missing,dimpaths.DimPath(dim),subtypes)
        return Param(slice.name, nstype)#}}}
uniquesig = UniqueSignature("unique")

class Argunique(UnaryFuncAggregateOp):
    _sigs = [uniquesig]

    @classmethod
    def _prepareSlice(self,slice):
        return ops.ensure_frozen(slice)

class CorrSignature(FuncSignature):
    def check(self, slice):
        if(len(slice.dims) < 2):
            return False

        if(not isinstance(slice.type,rtypes.TypeNumber)):
            return False

        corrdim = slice.dims[-2]
        corrdim2 = corrdim.insertDepDim(0,corrdim)
        ndims = dimpaths.DimPath(corrdim, corrdim2)

        ntype = rtypes.TypeReal64(slice.type.has_missing)
        ntype = dimpaths.dimsToArrays(ndims, ntype)

        slice = ops.PackArrayOp(slice,2)
        nkwargs = {'slice':slice}
        return (nkwargs, Param(slice.name, ntype))
corrsig = CorrSignature("corr")

class Corr(UnaryFuncElemOp):
    _sigs = [corrsig]

    def _finishSlice(self,slice):
        return ops.UnpackArrayOp(slice,2)

        
