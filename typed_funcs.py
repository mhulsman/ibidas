from collections import defaultdict
import numpy

class FuncSignature(object):
    __slots__ = ['types', 'impl']

    def __init__(self, types, impl):
        self.types = types
        self.impl = impl

func_info = defaultdict(list)


def add_func(names, types, impl):
    assert isinstance(types, tuple), "Types arguments should be a tuple"
    assert all(issubclass(t, rtypes.TypeUnknown) for t in types), "Type tuple should contain type classes"
    
    func_sig = FuncSignature(types, impl)
    for name in names:
        func_info[name].append(func_sig)



def find_impl(funcname, types, otype=None):
    assert isinstance(types, tuple), "Types arguments should be a tuple"
    assert all(isinstance(t, rtypes.TypeUnknown) for t in types), "Type tuple should contain type objects"

    sigs = func_info[funcname]
    typesmro = [t.__class__.__mro__ for t in types]
    
    min_sig = None
    min_dist = numpy.inf
    min_pos = []
    min_otype = None
    for sig in sigs:
        if not len(sig.types) == len(types):
            continue
        try:
            pos = [tmro.index(stypecls) for stypecls, tmro in zip(sig.types, typesmro)]
        except ValueError:
            continue
        rotype = sig.check(types, otype)
        if(otype is False):
            continue
        dist = sum(pos)
        if(dist < min_dist or (dist == min_dist and pos < min_pos)):
            min_dist = dist
            min_sig = sig
            min_pos = pos
            min_otype = rotype
                
    if(min_sig is None):
        if(otype is None):
            so = "*"
        else:
            so = str(otype)
        raise RuntimeError, "Cannot find func " + str(funcname)  + " with signature " + str(types) + " --> " + so
    
    return (min_sig.impl, min_otype)


class FuncImpl(object):
    @classmethod
    def check(self, types, otype):
        pass

class BiFuncImpl(object):
    @classmethod
    def check(self, types, otype):
        pass

    def reduce(self, seq):
        pass

class Func(object):
    def __init__(self, name):
        self.name = name

    def execute(self, source, *params, **kwds):
        pass
    __call__ = execute

class BiFunc(Func):
    def __init__(self, name, **kwds):
        self.left = None
        self.kwds = kwds
        Func.__init__(self, name, *params, **kwds)


    def copy(self):
        return copy.copy(self)

    def __or__(self, other):
        assert not self.left is None, "Error in use of operands"
        return self.execute(self.left, other, **self.kwds)

    def __ror__(self, other):
        res = self.copy()
        res.left = other
        return res

    def __call__(self, *params, **kwds):
        if(params or 'left' in kwds or 'right' in kwds):
            kwds.upddate(self.kwds)
            return self.execute(*params, **kwds)
        else:
            kwds.upddate(self.kwds)
            return self.__class__(self.name, **kwds)


    def execute(self, left, right, *params, **kwds):
        pass
