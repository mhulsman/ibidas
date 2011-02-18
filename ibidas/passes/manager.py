import time, math
from ..utils import topological_sort, util

class Pass(topological_sort.TopologicalSortable):
    """Pass object. Each pass should inherit from
    this object, and implement a new version of
    the method run."""

    @classmethod
    def run(cls, query, pass_results):
        """Should be reimplemented by child classes. 
        :param query: reference to query context
        :param pass_results: dictionary with results from earlier 
                        passes (accessible by using class as key).

        :rtype: Can return several values:
            - None
            - Object : stored in pass_results
            - (Object, invalidate_passes)
            - (Object, invalidate_passes, add_passes)
        
        """
class PassManager(object):
    """Class to handle passes. Passes are objects which 
    perform some function on a query object."""

    def __init__(self, log=False):
        """Initializes pass manager"""
        self.objs = []
        self.params = []
        self.log = log

    def register(self, objcls, *args, **kwargs):
        """Registers a new pass.
        Passes are ordered according to their topological
        characteristics. If no conflict arises, passes are kept
        in place of their register position. 
        """
        assert issubclass(objcls, Pass), \
            "Only pass classes can be registered"
        self.objs.append(objcls)
        self.params.append((args,kwargs))

    def run(self, query):
        """Performs registered passes on query, by
        performing a stable topological sort."""

        return PassManagerRun(self, query, self.log).run()


class PassManagerRun(object):
    def __init__(self,manager, query, log):
        self.query = query
        self.topoiter = topological_sort.topo_sorted(manager.objs,return_index=True)
        self.pass_results = {}
        self.objects = list(manager.objs)
        self.params = list(manager.params)
        self.log = log


    def invalidate(self, ipass):
        self.topoiter.invalidate(ipass)
        self.pass_results.pop(ipass)

    def add_pass(self, ipass, *params, **kwds):
        self.topoiter.prepend(ipass)
        self.objects.append(ipass)
        self.params.append((params,kwds))

    def run(self):
        if self.log:
            logtable = []
            totalstart = time.time()
        for cur_pass_idx in self.topoiter:
            cur_pass = self.objects[cur_pass_idx]
            param_args, param_kwds = self.params[cur_pass_idx]

            if(self.log):
                start = time.time()
            res = cur_pass.run(self.query, self, *param_args, **param_kwds)
            if(self.log):
                rtime = time.time() - start
                logtable.append((cur_pass.__name__, rtime))
            if(not res is None):
                self.pass_results[cur_pass] = res
                result = res
        if(self.log):
            totaltime = time.time() - totalstart
            restable = []
            for name, rtime in logtable:
                restable.append((name, util.format_runtime(rtime), u"(%.2f%%)" % (100 * (rtime/totaltime)),))
            print util.create_strtable(util.transpose_table(restable))
            print "-------------"
            print u"Total : %s" % (util.format_runtime(totaltime))

        return result
