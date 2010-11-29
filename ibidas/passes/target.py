from collections import defaultdict

import manager
_delay_import_(globals(),"..repops")

class TargetCalc(manager.Pass):
    """Calculates for representor objects in query tree
    the targets (i.e. other representor objects that use their data)"""

    @classmethod
    def run(cls, query, pass_results):
        target_dict = defaultdict(list)
        
        to_process = set([query.root])
        visited = set()

        while(to_process):
            rep = to_process.pop()
            if(isinstance(rep, repops.OpRep)):
                for source in rep._sources:
                    target_dict[source].append(rep)
                to_process.update(set(rep._sources) - visited)
            visited.add(rep)
       
        return target_dict
