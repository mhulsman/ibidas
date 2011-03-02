_delay_import_(globals(),"..representor")
_delay_import_(globals(),"util")

#after receipt by Ferdinand Jamitzky on activestate.com: http://code.activestate.com/recipes/384122-infix-operators/
class Infix:
    def __init__(self, function, params = (), kwargs={}):
        self.function = function
        self.kwargs = kwargs
        self.params = params

    def __ror__(self, other):
        return Infix(lambda x, y, z, self=self, other=other: self.function(other, x, *y, **z), self.params, self.kwargs)

    def __or__(self, other):
        return self.function(other, self.params, self.kwargs)

    def __call__(self, *params, **kwargs):
        kwargs.update(self.kwargs)
        params = self.params + params
        
        if(params and isinstance(params[0],representor.Representor)):
            return self.function(*params, **kwargs)
        else:
            return Infix(self.function,params,kwargs)
            
