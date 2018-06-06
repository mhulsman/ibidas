import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def update(self, seq):
        for pos in range(len(seq)):
            if seq[pos] in self.map:
                continue
            
            key = seq[pos]

            
            xpos = pos
            for xpos in range(pos+1, len(seq)):
                if seq[xpos] in self.map:
                    break

            if seq[xpos] in self.map:
                #insert before
                hooknode = self.map[seq[xpos]]
                ckey,cprev, cnext = hooknode
                hooknode[1] = cprev[2] = self.map[key] = [key, cprev, hooknode]
                #self.add(seq[pos])
            else:
                self.add(seq[pos])

    def copy(self):
        return OrderedSet(self)

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

class NamedList(list):
    def __init__(self, name, *args):
        list.__init__(self, *args)
        self.name = name


