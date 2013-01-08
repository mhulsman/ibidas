NF_ELSE=1
NF_ERROR=2
NF_ROBJ=4
F_CACHE=8


class VisitorBase(object):
    def __visitkey__(self, visited):
        return visited.__class__

    def __notfound__(self, prefix, visited, flags):
        xflag = flags & (NF_ROBJ | NF_ELSE | NF_ERROR)
        if xflag == NF_ROBJ:
            return None
        elif xflag == NF_ELSE:
            return getattr(self, prefix + "else")
        else:
            tmp = repr(visited.__class__)
            raise TypeError, ('No suitable visit method found for %s, %s' % (prefix, tmp))

    def __findmethod__(self, prefix, visitkey):
        mro = visitkey.__mro__
        method = None
        for cur_class in mro:
            name = prefix + cur_class.__name__
            try:
                method = getattr(self, name)
                break
            except AttributeError, e:
                pass
        return method

    def __visit__(self, prefix, flags, visited, visitkey, args, kwargs):
        cache = self.__dict__
        if flags & F_CACHE:
            cachekey = (prefix, visited)
            if cachekey in cache:
                return cache[cachekey]
        
        key = (prefix, visitkey)
        visit_method = cache.get(key, -1)
        if visit_method == -1:
            visit_method = self.__findmethod__(prefix, visitkey) 
            if visit_method is None:
                visit_method = self.__notfound__(prefix, visited, flags)
            cache[key] = visit_method

        if visit_method is None:
            return visited
       
        res = visit_method(visited, *args, **kwargs)

        if flags & F_CACHE:
            cache[cachekey] = res
        return res
                
class VisitorBaseDirect(VisitorBase):
    def __visitkeydirect__(self, visited):
        return visited

    def __findmethoddirect__(self, prefix, visitkey):
        name = prefix + str(visitkey)
        return getattr(self, name, None)

    def __visitdirect__(self, prefix, flags, visited, visitkey, args, kwargs):
        cache = self.__dict__
        if flags & F_CACHE:
            cachekey = (prefix, visited)
            if cachekey in cache:
                return cache[cachekey]
        
        key = (prefix, visitkey)
        visit_method = cache.get(key, -1)
        if visit_method == -1:
            visit_method = self.__findmethoddirect__(prefix, visitkey) 
            if visit_method is None:
                visit_method = self.__notfound__(prefix, visited, flags)
            cache[key] = visit_method

        if visit_method is None:
            return visited
       
        res = visit_method(visited, *args, **kwargs)

        if flags & F_CACHE:
            cache[cachekey] = res
        return res


def DirectVisitorFactory(name='Visitor', prefixes=('visit',), bases=(), flags=0):
    return VisitorFactory(name, prefixes, bases, flags, direct=True)

def genFunctions(prefix, flags):
    def visit(self, *args, **kwargs):
        if not args:
            raise TypeError, "Visit method should be called with the object to visit"
        key = self.__visitkey__(args[0])
        return self.__visit__(prefix, flags, args[0], key, args[1:], kwargs)

    def visitKey(self, *args, **kwargs):
        if len(args) <= 1:
            raise TypeError, "Visit method should be called with the object to visit"

        return self.__visit__(prefix, flags, args[1], args[0], args[2:], kwargs)
    return (visit, visitKey)

def genFunctionsDirect(prefix, flags):
    def visit(self, *args, **kwargs):
        if not args:
            raise TypeError, "Visit method should be called with the object to visit"
        key = self.__visitkeydirect__(args[0])
        return self.__visitdirect__(prefix, flags, args[0], key, args[1:], kwargs)

    def visitKey(self, *args, **kwargs):
        if len(args) <= 1:
            raise TypeError, "Visit method should be called with the object to visit"

        return self.__visitdirect__(prefix, flags, args[1], args[0], args[2:], kwargs)
    return (visit, visitKey)

def VisitorFactory(name='Visitor', prefixes=("visit",), bases=(), flags=0, direct=False):
    if direct:
        bases = (VisitorBaseDirect,) + bases
    else:
        bases = (VisitorBase,) + bases
    ntype = type(name, bases, {})

    for prefix in prefixes:
       if direct:
           visit,visitKey = genFunctionsDirect(prefix, flags)
       else:
           visit,visitKey = genFunctions(prefix, flags)
       setattr(ntype, prefix, visit)
       setattr(ntype, prefix + "Key", visitKey)

    return ntype       
