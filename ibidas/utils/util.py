import numpy
import hashlib
from itertools import izip
import operator
import collections
try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except ImportError:
    try:
        from IPython.core.debugger import Tracer; debug_here = Tracer()
    except:
        pass
import cPickle, zlib
import math
import os.path
import gc, sys
import itertools
import csv
import cutils


#objs: objs to find names for
#req_objs: objs which should be in same frame
def find_names(objs, req_objs=[],skip_frames=True):
    #after: http://pythonic.pocoo.org/2009/5/30/finding-objects-names
    result = []
    referrers = dict()
    refsetobjs = []

    frame = sys._getframe()
    #refs should be in locals dict
    frameobj = []
    for fcounter,frame in enumerate(iter(lambda: frame.f_back, None)):
        fid = id(frame.f_locals)
        if not skip_frames or fcounter >= 1:
            referrers[fid] = frame.f_locals
            frameobj.append(fid)
    refsetobjs.append(set(frameobj))

    for obj in itertools.chain(objs,req_objs):
        refsetobj = [id(referrer) for referrer in gc.get_referrers(obj) if isinstance(referrer, dict)]
        refsetobjs.append(set(refsetobj))
    refs_with_all_objs = reduce(operator.__and__,refsetobjs)

    for refid in frameobj:
        if not refid in refs_with_all_objs:
            continue
        referrer = referrers[refid]
        result = []
        for obj in objs:
            for k, v in referrer.iteritems():
                if v is obj and not k.startswith('_') and not k[:2].isupper():
                    result.append(k)
        if(len(result) == len(objs)):
            return result
    return []

_delay_import_(globals(),"missing","Missing")
def save_rep(r, filename):
    f = open(filename, 'wb')
    s = cPickle.dumps(r,protocol=2)
    s = zlib.compress(s)
    f.write(s)


def save_csv(r, filename):
    f = open(filename,'wb')
    r= r.Array(tolevel=1)
    data = r.Tuple().ToPython();
    if filename.endswith('tsv'):
        w = csv.writer(f,delimiter='\t');
    else:
        w = csv.writer(f);
    w.writerow(r.Names);
    w.writerows(data);

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
    filename = os.path.expanduser(filename)
    if(filename.endswith("gz")):
        import gzip
        file = gzip.open(filename)
    else:
        file = open(filename)
    return file


#after IPython implementation
units = [u"s", u"ms",u'us',"ns"]
scaling = [1, 1e3, 1e6, 1e9]

def format_runtime(rtime):
    if rtime > 0.0 and rtime < 1000.0:
        order = min(-int(math.floor(math.log10(rtime)) // 3), 3)
    elif rtime >= 1000.0:
        order = 0
    else:
        order = 3
    return u"%.*g %s" % (3, rtime * scaling[order],units[order])


def check_module(modulename):
    try:
        __import__(modulename)
        return True
    except ImportError:
        return False

#from: http://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
class memoized(object):
   """Decorator that caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned, and
   not re-evaluated.
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
      self.__name__ = func.__name__
   def __call__(self, *args, **kwargs):
       if kwargs: #uncacheable
            return self.func(*args, **kwargs)
       try:
          return self.cache[args]
       except KeyError:
          value = self.func(*args)
          self.cache[args] = value
          return value
       except TypeError:
          # uncachable -- for instance, passing a list as an argument.
          # Better to not cache than to blow up entirely.
          return self.func(*args)

   def __repr__(self):
       """Return the function's docstring."""
       return self.func.__doc__

   def __get__(self, obj, objtype):
       """Support instance methods."""
       return functools.partial(self.__call__, obj)

def check_realseq(data):
    return isinstance(data,collections.Sequence) and not isinstance(data, basestring)

def get_shape(data, lengths, pos=0):
    if check_realseq(data):
        if len(lengths) > pos + 1 and len(lengths[pos + 1]) <= 1:
            for elem in data:
                if not get_shape(elem,lengths,pos+1):
                    break
        lengths[pos].add(len(data))
    else:
        lengths[pos].add(0)
    return len(lengths[pos]) == 1

def fill(data, x, pos):
    if pos > 1:
        for i in range(x.shape[0]):
            fill(data[i],x[i,...],pos-1)
    else:
        x[:] = data

def replace_darray(data, type=object, maxdim=1, mindim=0):
    try:
        return cutils.darray(data,type,maxdim,mindim)
    except ValueError:
        if maxdim == 1:
            if not check_realseq(data):
                if mindim == 0:  return data
                else: raise ValueError, "Object not deep enough"
            else:
                nlen = len(data)
                x = numpy.zeros((nlen,),dtype=type)
                x[:] = data
                return x
        #maxdim > 1        
        lengths = [set() for i in range(maxdim)]
        get_shape(data, lengths)
        cusedim = 0
        for lg in lengths:
            if len(lg) == 1:
                cusedim = cusedim + 1
        if cusedim < mindim:
            raise ValueError, "Object not deep enough"
        shape = [lengths[i].pop() for i in range(cusedim)]
        x = numpy.zeros(tuple(shape),dtype=type)
        fill(data,x, len(x.shape))
        return x
        

nversion = numpy.__version__.split('.')    
numpy16up = int(nversion[0]) >= 1 and int(nversion[1]) >= 6

if numpy16up:
    darray = replace_darray
else:
    darray = cutils.darray
    
