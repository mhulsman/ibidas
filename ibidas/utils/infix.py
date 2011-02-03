#after receipt by Ferdinand Jamitzky on activestate.com: http://code.activestate.com/recipes/384122-infix-operators/

class Infix:
    def __init__(self, function, kwargs={}):
        self.function = function
        self.kwargs = kwargs

    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x), self.kwargs)

    def __or__(self, other):
        return self.function(other,**self.kwargs)

    def __call__(self, *params, **kwargs):
        if(params):
            kwargs.update(self.kwargs)
            return self.function(*params, **kwargs)
        else:
            kwargs.update(self.kwargs)
            return Infix(self.function,kwargs)
            

class RevInfix:
    def __init__(self, function, kwargs={}):
        self.function = function
        self.kwargs = kwargs

    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(x, other), self.kwargs)

    def __or__(self, other):
        return self.function(other,**self.kwargs)

    def __call__(self, *params, **kwargs):
        if(params):
            kwargs.update(self.kwargs)
            left,right = params[:2]
            params = params[2:]
            return self.function(right,left, *params, **kwargs)
        else:
            kwargs.update(self.kwargs)
            return Infix(self.function,kwargs)

