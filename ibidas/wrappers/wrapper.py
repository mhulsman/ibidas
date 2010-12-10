from .. import representor
from ..passes import manager

class Indicator(manager.Pass):
    def __eq__(self, other):
        return self.__class__ is other.__class__
    
    def __hash__(self):
        return hash(self.__class__)

class Executor(manager.Pass):
    pass


class SourceRepresentor(representor.Representor):
    pass

def open(func, *args, **kwds):
    return func(*args, **kwds)
