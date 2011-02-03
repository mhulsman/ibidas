_delay_import_(globals(),"repops")
from utils import util

class QueryContext(object):
    """Context object of a query. Encloses representor
    with fixate operation."""
    def __init__(self, representor, args={}, endpoint=True):
        if(endpoint):
            self.root = repops.Fixate(representor)
        else:
            self.root = repops.Gather(representor)
            
        self.args = args

