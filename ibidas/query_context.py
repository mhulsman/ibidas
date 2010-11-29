_delay_import_(globals(),"repops")

class QueryContext(object):
    """Context object of a query. Encloses representor
    with fixate operation."""
    def __init__(self, representor, args={}):
        self.root = repops.fixate(representor)
        self.args = args

