from ..utils import topological_sort

class Pass(topological_sort.TopologicalSortable):
    """Pass object. Each pass should inherit from
    this object, and implement a new version of
    the method run."""

    @classmethod
    def run(cls, query, pass_results):
        """Should be reimplemented by child classes. 
        Parameters
        ----------
        query: reference to query context
        pass_results: dictionary with results from earlier 
                        passes (accessible by using class as key).

        Returns
        -------
        - None
        - Object : stored in pass_results
        - (Object, invalidate_passes)
        - (Object, invalidate_passes, add_passes)
        
        """
class PassManager(object):
    """Class to handle passes. Passes are objects which 
    perform some function on a query object."""

    def __init__(self):
        """Initializes pass manager"""
        self.objs = []

    def register(self, objcls):
        """Registers a new pass.
        Passes are ordered according to their topological
        characteristics. If no conflict arises, passes are kept
        in place of their register position. 
        """
        assert issubclass(objcls, Pass), \
            "Only pass classes can be registered"
        self.objs.append(objcls)

    def run(self, query):
        """Performs registered passes on query, by
        performing a stable topological sort."""

        topoiter = topological_sort.topo_sorted(self.objs)
        pass_results = {}

        for cur_pass in topoiter:
            res = cur_pass.run(query, pass_results)
            
            if(not isinstance(res, tuple)):
                result = res
                pass_results[cur_pass] = res
                continue
            if(len(res) > 0):
                result = res[0]
                pass_results[cur_pass] = res[0]
            if(len(res) > 1):
                for invalidate in res[1]:
                    topoiter.invalidate(invalidate)
                    if invalidate in query.pass_results:
                        del query.pass_results[invalidate]

            if(len(res) > 2):
                for add_pass in res[2]:
                    topoiter.append(add_pass)

        return result
