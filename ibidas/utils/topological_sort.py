from collections import defaultdict

class TopologicalSortable(object):
    after = set()
    before = set()


class topo_sorted(object):
    def __init__(self, objiter):
        self.objs = list(objiter)
        self.retcls = set()
        self.availcls = defaultdict(int)

        self.before_cls = defaultdict(set)

        for obj in self.objs:
            if(self.availcls[obj.__class__] == 0):
                self.before_cls[obj.__class__].update(obj.before)
            self.availcls[obj.__class__] += 1

   
    def invalidate(self, cls):
        self.retcls.discard(cls)

    def _addobj(self, obj):
        if(len(obj.before & self.retcls) > 0):
            raise RuntimeError, "Cannot add " + str(obj) + \
                " as before constraint is violated by: " + \
                    str(obj.before & self.retcls)

        if(self.availcls[obj.__class__] == 0):
            self.before_cls[obj.__class__].add(obj.before)
        
        self.availcls[obj.__class__] += 1
    
    def __contains__(self, obj):
        if(issubclass(obj, TopologicalSortable)):
            return self.availcls[obj] > 0
        else:
            return obj in self.objs

    def append(self, obj):
        self._addobj(obj)
        self.objs.append(obj)

    def prepend(self, obj):
        self._addobj(obj)
        self.objs.insert(obj, 0)
    
    def __iter__(self):
        return self

    def next(self):
        for pos, objcls in enumerate(self.objs):
            #after constraints fullfilled?
            if(not objcls.after.issubset(self.retcls)):
                continue

            #are there objects that want to run before me?
            if(sum(self.availcls[before] 
                    for before in self.before_cls[objcls])):
                continue
            
            del self.objs[pos]
            self.availcls[objcls] -= 1
            self.retcls.add(objcls)
            return objcls
        
        if(not self.objs):
            raise StopIteration
        
        raise RuntimeError, "Cycle in topological sort: " + \
                                    "cannot return next object"



  
