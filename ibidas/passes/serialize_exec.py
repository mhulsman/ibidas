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
        params = dict()
        for edge in self.graph.edge_target[node]:
            source = edge.source
            if(source in self.graph.na['exec_order']):
                param_idx = self.graph.na['exec_order'][source]
            else:
                param_idx = self.visit(source)
            self.graph.na['param_usecount'][source] += 1
            
            if edge.type == "param":
                params[edge.subtype] = param_idx
            elif(edge.type == "paramlist"):
                if(not edge.subtype in params):
                    params[edge.subtype] = [None]
                while(len(params[edge.subtype]) <= edge.attr):
                    params.append(None)
                params[edge.subtype][edge.attr] = param_idx

        command_id = len(self.commands)
        self.commands.append(node)
        self.graph.na['exec_order'][node] = command_id
        self.graph.na['param_idxs'][node] = params
        self.graph.na['param_usecount'][node] = 0
        return command_id

