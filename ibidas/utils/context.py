import __builtin__

class Context(object):
    """Context objects store all actions on them, which
    can then be replayed on another object with the _apply function below"""

    __slots__ = ['_action','_prev']
    __aterror__ = set(('__array_struct__','__array__','__array_interface__',\
                        '__array_priority__','__array_wrap__','trait_names','_getAttributeNames'))

    def __init__(self,prev,*action):
            self._prev = prev
            self._action = action

    def __getattr__(self,name):
        #some support for numpy and ipython
        if(name in self.__aterror__):
            raise AttributeError
        r = Context(self,'DOT',name)
        return r

        #return Context(self,'DOT',name)

    def __getitem__(self,key):
        return Context(self,'ITEM',key)

    def __repr__(self):
        return repr(self._prev) + str(self._action)

    def __call__(self,*params,**kwds):
        return Context(self,'CALLSELF',params,kwds)
   
    def __iter__(self):
        return Context(self,"ITER")

    def next(self):
        return Context(self,"NEXT")

    def __neg__(self):
        return Context(self,'UNOP','__neg__')

    def __pos__(self):
        return Context(self,'UNOP','__pos__')
    
    def __invert__(self):
        return Context(self,'UNOP','__invert__')

    def __add__(self, other):
        return Context(self,'BINOP','__add__',other)

    def __radd__(self, other):
        return Context(self,'BINOP','__radd__',other)
    
    def __sub__(self, other):
        return Context(self,'BINOP','__sub__',other)

    def __rsub__(self, other):
        return Context(self,'BINOP','__rsub__',other)

    # multiplication ( * )
    def __mul__(self, other):
        return Context(self,'BINOP','__mul__',other)
    
    def __rmul__(self, other):
        return Context(self,'BINOP','__rmul__',other)

    # division ( / )
    def __div__(self, other):
        return Context(self,'BINOP','__div__',other)
    
    def __rdiv__(self, other):
        return Context(self,'BINOP','__rdiv__',other)
    
    # floor division ( // )
    def __floordiv__(self, other):
        return Context(self,'BINOP','__floordiv__',other)
    
    def __rfloordiv__(self, other):
        return Context(self,'BINOP','__rfloordiv__',other)
    
    # power (**)
    def __pow__(self, other):
        return Context(self,'BINOP','__pow__',other)
    
    def __rpow__(self, other):
        return Context(self,'BINOP','__rpow__',other)
    
    # modulo (%)
    def __mod__(self, other):
        return Context(self,'BINOP','__mod__',other)
    
    def __rmod__(self, other):
        return Context(self,'BINOP','__rmod__',other)
 
    # and operator ( & )
    def __and__(self, other):
        return Context(self,'BINOP','__and__',other)
    
    def __rand__(self, other):
        return Context(self,'BINOP','__rand__',other)
    
    # or operator ( | )
    def __or__(self, other):
        return Context(self,'BINOP','__or__',other)

    def __ror__(self, other):
        return Context(self,'BINOP','__ror__',other)
     
    # exclusive-or operator ( ^ )
    def __xor__(self, other):
        return Context(self,'BINOP','__xor__',other)

    def __rxor__(self, other):
        return Context(self,'BINOP','__rxor__',other)

    # less-than ( < )
    def __lt__(self, other):
        return Context(self,'BINOP','__lt__',other)

    # less-than-or-equals ( <= )
    def __le__(self, other):
        return Context(self,'BINOP','__le__',other)

    # equals ( == )
    def __eq__(self, other):
        return Context(self,'BINOP','__eq__',other)

    # not-equals ( != )
    def __ne__(self, other):
        return Context(self,'BINOP','__ne__',other)

    # greater-than ( > )
    def __gt__(self, other):
        return Context(self,'BINOP','__gt__',other)

    # greater-than-or-equals ( >= )
    def __ge__(self, other):
        return Context(self,'BINOP','__ge__',other)

    def _call(self, func, *params, **kwds):
        return Context(self, "APPLYFUNC", func, params, kwds)

def _apply(ct, obj, **extra_params):
    """Apply a context ct to an object obj"""
    result = obj
    actionlist = []
    while(not ct._prev is None):
        actionlist.append(ct._action)
        ct = ct._prev
   
    for action in actionlist[::-1]:
        if(action[0] == 'DOT'):
            result = getattr(result,action[1])
        elif(action[0] == 'ITEM'):
            result = result[action[1]]
        elif(action[0] == 'CALLSELF'):
            param = []
            for p in action[1]:
                #if(isinstance(p,Context)):
                #    param.append(_apply(p, obj, **extra_params))
                #else:
                param.append(p)
            kwds = dict()
            for k,v in action[2].iteritems():
                #if(isinstance(v,Context)):
                #    kwds[k] = _apply(v, obj, **extra_params)
                #else:
                kwds[k] = v

            if(extra_params):
                if(result.func_name in extra_params):
                    kwds.update(extra_params[result.func_name])
            result = result(*param, **kwds)
        elif(action[0] == 'BINOP'):
            if(isinstance(action[2],Context)):
                other = _apply(action[2], obj, **extra_params)
            else:
                other = action[2]
            result = getattr(result,action[1])(other)
        elif(action[0] == 'UNOP'):
            result = getattr(result,action[1])()
        elif(action[0] == 'SETITEM'): #cannot be recorded, but can be played
            if(isinstance(action[2],Context)):
                value = _apply(action[2], obj, **extra_params)
            else:
                value = action[2]
            result.__setitem__(action[1],action[2])
        elif(action[0] == 'APPLYFUNC'):
            param = []
            for p in action[2]:
                if(isinstance(p,Context)):
                    param.append(_apply(p, obj, **extra_params))
                else:
                    param.append(p)
            kwds = action[3].copy()
            for k,v in action[3].iteritems():
                if(isinstance(v,Context)):
                    kwds[k] = _apply(v, obj, **extra_params)
            if(extra_params):
                if(action[1].func_name in extra_params):
                    kwds.update(extra_params[action[1].func_name])
            result = action[1](result,*param, **kwds)
    return result


_=Context(None)
__builtin__.__dict__['_'] = _
