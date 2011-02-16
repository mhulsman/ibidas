import numpy
import hashlib
from itertools import izip
import operator
import collections
from IPython.Debugger import Tracer; debug_here = Tracer()
import cPickle, zlib

_delay_import_(globals(),"missing","Missing")
def save_rep(r, filename):
    f = open(filename, 'wb')
    s = cPickle.dumps(r,protocol=2)
    s = zlib.compress(s)
    f.write(s)

def load_rep(filename):
    f = open(filename, 'rb')
    s = zlib.decompress(f.read())
    return cPickle.loads(s)

def valid_name(name):
    name = name.lower()
    newname = []
    for char in name:
        if(char.isalnum() or char == "_"):
            newname.append(char)
        else:
            newname.append("_")
    newname = "".join(newname)
    newname = newname.strip("_")
    if(not newname):
        return "funknown"
    if(newname[0].isdigit()):
        newname = "_" + newname
    return str(newname)
        

def seqgen(start=0):
    """Generator, generating monotonically increasing indexes"""
    while True:
        start += 1
        yield start


def unique(seq):
    """Given a list or tuple, returns a list or tuple in which each element
    occurs only once, while keeping the original order or the list"""
    seen_elems = set()
    res = [elem for elem in seq 
        if not elem in seen_elems and 
        not seen_elems.add(elem)]

    if(isinstance(seq, tuple)):
        return tuple(res)
    else:
        return res

def select(seq, index):
    """Given list or tuple in `seq`, select using `index`, 
    and returns list or tuple.

    :param seq: list or tuple
    :param index: int, long, sequence of indexes, sequence of booleans
    """
    if(isinstance(index, slice)):
        return seq[index]
    elif(operator.isSequenceType(index)) :
        if(len(index) == 0):
            return seq.__class__()

        if(isinstance(index[0], bool)):
            res = [elem for elem, indicator in zip(seq, index) if indicator]
        else:
            res = [seq[idx] for idx in index]
        
        if(isinstance(seq, tuple)):
            return tuple(res)
        return res

    if(isinstance(seq, tuple)):
        return (seq[index],)
    else:
        return [seq[index]]


def index_field(seq, fieldname):
    """Given a sequence in `seq`, retrieves attribute 
    with name `fieldname` from all
    elements, and creates a dictionary from fieldname value to element.
    
    :param seq: sequence of objects
    :param fieldname: attribute available in each object
    """
    func = operator.attrgetter(fieldname)
    return dict((func(elem), elem) for elem in seq)


def replace_in_tuple(tpl, replace):
    """Replaces objects in `tpl`, by using dictionary in `replace`"""
    return tuple([replace.get(elem, elem) for elem in tpl])

def delete_from_tuple(tpl, remove):
    return tuple([elem for elem in tpl if not elem in remove])


def zip_broadcast(*elems):
    lengths = set([len(elem) for elem in elems])
    lengths.discard(1)
    assert len(lengths) <= 1, "Number of elements in zip broadcast should be equal (or 1)"
    if(lengths):
        length = lengths.pop()
    else:
        length = 1
  
    elems = list(elems)
    for pos, elem in enumerate(elems):
        if(len(elem) == 1):
            elems[pos] = elem * length
    return zip(*elems)

def zip_broadcast_old(*elems):
    iters = []
    firstres = []
    for elem in elems:
        if(isinstance(elem, collections.Iterable)):
            iters.append(iter(elem))
            firstres.append(iters[-1].next())
        else:
            iters.append(None)
            firstres.append(elem)

    yield firstres

    secondres = []
    for pos, elem in enumerate(iters):
        if(elem is None):
            secondres.append(firstres[pos])
        else:
            try:
                secondres.append(elem.next())        
            except StopIteration:
                secondres.append(firstres[pos])
                iters[pos] = None

    if(all(elem is None for elem in iters)):
        raise StopIteration

    yield tuple(secondres)

    idxs = range(len(iters))
    while True: 
        res = [None] * len(iters)
        for pos in idxs:
            elem = iters[pos]
            if(not elem):
                res[pos] = firstres[pos]
            else:
                res[pos] = elem.next()
        yield res

def filter_missing(func):
    def fmissing(x, *params):
        if(x is Missing):
            return x
        else:
            return func(x, *params)
    return fmissing

def create_strtable(table):
    column_width = []
    for col in table:
        col_width = max([len(row) for row in col])
        column_width.append(col_width)

    res = ""
    for idx in range(len(table[0])):
        for cwidth, col in zip(column_width, table):
            res += col[idx].ljust(cwidth + 1)
        res += "\n"
    return res

def transpose_table(table):
    ncols = len(table[0])
    cols = []
    for col in range(ncols):
        cols.append([row[col] for row in table])
    return cols
        


def contained_in(inner_tuple, outer_tuple):
    pos = 0
    linner_tuple = len(inner_tuple)
    louter_tuple = len(outer_tuple)

    if(linner_tuple > louter_tuple):
        return False
    if(not inner_tuple):
        return (0,0)

    try:
        first_pos = pos = outer_tuple.index(inner_tuple[0])
    except ValueError, e:
        return False

    for elem in inner_tuple[1:]:
        try:
            pos = outer_tuple.index(elem, pos)
        except ValueError, e:
            return False

    return (first_pos, pos + 1)
    

class farray(numpy.ndarray):
    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                   self.shape == other.shape and
             numpy.equal(self.view(numpy.ndarray),other.view(numpy.ndarray)).all())

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        if(len(self) != len(other)):
            return len(self) > len(other)
        else:
            for x, y in izip(self, other): 
                if(x == y):
                    continue
                return x > y
            else:
                return False

    def __ge__(self, other):
        if(len(self) != len(other)):
            return len(self) >= len(other)
        else:
            for x, y in izip(self, other): 
                if(x == y):
                    continue
                return x >= y
            else:
                return True

    def __lt__(self, other):
        if(len(self) != len(other)):
            return len(self) < len(other)
        else:
            for x, y in izip(self, other): 
                if(x == y):
                    continue
                return x < y
            else:
                return False
        return 

    def __le__(self, other):
        if(len(self) != len(other)):
            return len(self) <= len(other)
        else:
            for x, y in izip(self, other): 
                if(x == y):
                    continue
                return x <= y
            else:
                return True
        return 

    def __hash__(self):
        res = hash(self.shape) ^ hash(self.dtype)
        if(self.dtype == object):
            for elem in self.ravel():
                res ^= hash(elem)
        else:
            res ^= hash(hashlib.md5(self.data).hexdigest())
        return res


def open_file(filename):
    if(filename.endswith("gz")):
        import gzip
        file = gzip.open(filename)
    else:
        file = open(filename)
    return file
