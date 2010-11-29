from collections import defaultdict

import manager
import prewalk

from ..utils.multi_visitor import VisitorFactory, NF_ELSE

class PlanWrapper(VisitorFactory(prefixes=("require", "func"), 
                                      flags=NF_ELSE), 
                       manager.Pass):
    @classmethod
    def run(cls, query, pass_results):
        return
        self = cls()
        prewalk = pass_results[prewalk.PreOrderWalk]

        self.indicator_attr = '_' + query.type + "_indicator"
        
        node_wrapper = {}
        self.queue = list(prewalk)
        while(self.queue):
            node = self.queue.pop()
            self.plan(node, node_wrapper)
    
    
    def planSourceRepresentor(self, node, node_wrapper):
        self.node_wrapper[node] = getattr(node, self.indicator_attr)

    def planUnaryOpRep(self, node, node_wrapper):
        w = node_wrapper[node._sources[0]]
        if(not w.compatible(node, self)):
            nnode = w.insertExitNode(node)
            self.queue.append(nnode)
        else:
            w.eat(node, self)

    def planMultiOpRep(self, node, node_wrapper):
        swrappers = [node_wrapper[snode] for snode in node._sources[0]]
        wpos_score = []
        wpos = []
        for w in set(swrappers):
            res = w.compatible(node, self)
            if(res):
                wpos_score.append((res, w))
        if(len(wpos_score) > 1):
            getfunc = operator.itemgetter(0)
            w = max(wpos_score, key=getfunc)[1]
        elif(len(wpos_score) == 1):
            w = wpos_score[0][1]
        else:
            pass    

        w.eat(node, self)

                
            

