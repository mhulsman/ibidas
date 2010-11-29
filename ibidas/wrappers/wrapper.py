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
    def request_ids(self, req_ids, source_ids):
        self._req_ids = req_ids

def open(func, *args, **kwds):
    return func(*args, **kwds)
