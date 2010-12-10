import operator

import manager
import create_graph
from ..utils.multi_visitor import VisitorFactory, NF_ELSE

class SerializeExec(VisitorFactory(prefixes=("visit",), flags=NF_ELSE), manager.Pass):

    after = set([create_graph.CreateGraph])

    @classmethod
    def run(cls, query, run_manager):
        self = cls()
        self.graph = run_manager.pass_results[create_graph.CreateGraph]
        self.commands = []
        self.visit(self.graph.root)
        return self.commands
    
    def visitelse(self,node):
        raise RuntimeError, "Unknown type of node encountered: " + str(type(node))

    def visitNode(self,node):
        source_edges = [edge for edge in self.graph.edge_target[node] if edge.type == "param"]
        source_edges.sort(key=operator.attrgetter("attr"))

        param_kwds = dict()
        param_args = []
        for source_edge in source_edges:
            source = source_edge.source
            if(source in self.graph.na['exec_order']):
                param_idx = self.graph.na['exec_order'][source]
            else:
                param_idx = self.visit(source)
            self.graph.na['param_usecount'][source] += 1
            if(isinstance(source_edge.attr,int)):
                assert source_edge.attr == len(param_args), "Missing argument"
                param_args.append(param_idx)
            else:
                param_kwds[source_edge.attr] = param_idx

        command_id = len(self.commands)
        self.commands.append(node)
        self.graph.na['exec_order'][node] = command_id
        self.graph.na['param_idxs'][node] = (param_args,param_kwds)
        self.graph.na['param_usecount'][node] = 0
        return command_id

