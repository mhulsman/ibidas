import manager
import target
_delay_import_(globals(),"ibidas","repops")


class PreOrderWalk(manager.Pass):
    """Determines a parent -->children order on the query tree (where
    parent is the root node, and children are its sources).
    Of course, order can easily be reversed."""
    after = set([target.TargetCalc])

    @classmethod
    def run(cls, query, pass_results):
        target_dict = pass_results[target.TargetCalc]

        target_counter = {}
        for k, v in target_dict.iteritems():
            target_counter[k] = len(v)
       
        prewalk = [query.root]
        pos = 0
        while(pos < len(prewalk)):
            rep = prewalk[pos]
            pos += 1
            if(isinstance(rep, repops.OpRep)):
                for source in rep._sources:
                    target_counter[source] -= 1
                    if(not target_counter[source]):
                        prewalk.append(source)
        assert len(target_dict) +1 == len(prewalk), \
                "Determining pre walk order failed. Cycle?"
        
        return prewalk#}}}

