import operator

import manager
import create_graph
from ..utils.multi_visitor import VisitorFactory, NF_ELSE
from ..utils import util
_delay_import_(globals(),"..query_graph")
class SerializeExec(VisitorFactory(prefixes=("visit","assign"), flags=NF_ELSE), manager.Pass):

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
        params = dict()
        for edge in self.graph.edge_target[node]:
            if(isinstance(edge, query_graph.ParamEdge)):
                source = edge.source
                if(source in self.graph.na['exec_order']):
                    param_idx = self.graph.na['exec_order'][source]
                else:
                    param_idx = self.visit(source)
                self.graph.na['param_usecount'][source] += 1

                self.assign(edge, params, param_idx)
               

        command_id = len(self.commands)
        self.commands.append(node)
        self.graph.na['exec_order'][node] = command_id
        self.graph.na['param_idxs'][node] = params
        self.graph.na['param_usecount'][node] = 0
        return command_id

    def assignParamEdge(self, edge, params, param_idx):
        params[edge.name] = param_idx

    def assignParamListEdge(self, edge, params, param_idx):
        if(not edge.name in params):
            params[edge.name] = [None]
        while(len(params[edge.name]) <= edge.pos):
            params[edge.name].append(None)
        params[edge.name][edge.pos] = param_idx


