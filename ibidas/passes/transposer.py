import itertools
import manager
import create_graph
import python_peephole
import serialize_exec

from ..utils import util
from ..utils.multi_visitor import VisitorFactory, NF_ERROR, F_CACHE
from ..constants import *
_delay_import_(globals(),"..query_graph")
_delay_import_(globals(),"..ops")
_delay_import_(globals(),"..itypes","dimpaths","rtypes")
_delay_import_(globals(),"..repops_funcs")

class Transposer(VisitorFactory(prefixes=("visit",), flags=NF_ERROR), manager.Pass):
    after=set([python_peephole.PythonPeepHole])
    before=set([serialize_exec.SerializeExec])

    @classmethod
    def run(cls, query, run_manager):
        self = cls()
        self.graph = run_manager.pass_results[create_graph.CreateGraph]

        for node in list(self.graph.nodes):
            self.visit(node)
    
    def visitNode(self, node):
        pass

    def visitFilterOp(self, node):

        #check 1: all dims fixed, more than one dim
        if not len(node.dims) > 1:
            return
        if not all([not d.isVariable() for d in node.dims]):
            return

        #check 2: type dim dependent
        if not node.type.__class__ is rtypes.TypeArray:
            return

        d = node.type.dims[0]
        if not d.isVariable():
            return

        #check 3: type dim not dependent on all dims
        if not (len(d.dependent) < len(node.dims) or not all(d.dependent)):
            return

        #check 5: type dim aggregated, single use, aggretation type does not include dim
        for n in self.graph.walkUseContig(node, include_first=False):
            if isinstance(n, (ops.UnpackArrayOp, ops.PackArrayOp)):
                pass
            elif isinstance(n, ops.UnaryFuncAggregateOp):
                last = self.graph.getDataEdge(n)
                packdim = last.source.dims[-n.packdepth]
                if not packdim == d:
                    return #FIXME: possible to have multiple unaryfuncaggregates...
                if n.type.__class__ == rtypes.TypeArray:
                    if n.type.dims[0] in node.dims:
                        return 
                aggnode = n
                break
            else:
                return
        else:#ran out of contig
            return

        #determine permute dims
        permdim = set()
        for dep, dim in itertools.izip_longest(d.dependent, node.dims[::-1], fillvalue=False):
            if not dep:
                permdim.add(dim)

        
        #insert pre permute
        constraint = self.graph.getDataEdge(node, name="constraint").source
        data = self.graph.getDataEdge(node).source
        


        #check 6: all constraint insert dims correspond to perm dims
        constraint_insert_dims = []
        tdims = list(node.dims)
        for n in self.graph.walkSourceContig(constraint):
                if isinstance(n, ops.InsertDimOp):
                    if not tdims[n.matchpoint] in permdim:
                        return
                    constraint_insert_dims.append(tdims[n.matchpoint])
                    del tdims[n.matchpoint]
                else:
                    constraint_source_node = n
                    assert tuple(tdims) == tuple(n.dims), "Dims do not match after deinstertion"
                    break
        else:
                return #ran out of contig
       
        #limit permdims to constraint insert dims
        permdim = permdim & set(constraint_insert_dims)

        #check 7: all data insert dims correspond to non perm dims
        data_insert_dims = []
        data_insert_dims_bc = []
        tdims = list(node.dims)
        for n in self.graph.walkSourceContig(data):
                if isinstance(n, ops.InsertDimOp):
                    if tdims[n.matchpoint] in permdim:
                        return
                    data_insert_dims.append(tdims[n.matchpoint])
                    data_insert_dims_bc.append(n.dims[n.matchpoint])
                    del tdims[n.matchpoint]
                else:
                    data_source_node = n
                    assert tuple(tdims) == tuple(n.dims), "Dims do not match after deinsertion"
                    break
        else:
                return #ran out of contig

        #add unpack node
        while(data_source_node.type.__class__ == rtypes.TypeArray and
                data_source_node.dims[0] in permdim):
                newnode = ops.UnpackArrayOp(data_source_node)
                self.graph.addNode(newnode)
                self.graph.addEdge(query_graph.ParamEdge(data_source_node, newnode,"slice"))
                data_source_node = newnode

        #determine permute_idxs in 
        before = []
        after = []
        for pos, dim in enumerate(data_source_node.dims):
            if dim in permdim:
                after.append(pos)
            else:
                before.append(pos)
        permute_idxs = before + after
        
        #add permute node
        newnode = ops.PermuteDimsOp(data_source_node, permute_idxs)
        self.graph.addNode(newnode)
        self.graph.addEdge(query_graph.ParamEdge(data_source_node, newnode,"slice"))
        data_source_node = newnode
                
        #add pack node of perm dims
        newnode = ops.PackArrayOp(data_source_node, len(after)+1)
        self.graph.addNode(newnode)
        self.graph.addEdge(query_graph.ParamEdge(data_source_node, newnode,"slice"))
        data_source_node = newnode
        
        #redo inserts
        insertpos = 0
        for dim in constraint_source_node.dims:
                if not dim in data_source_node.dims:
                    bcdim = data_insert_dims_bc[data_insert_dims.index(dim)]
                    newnode = ops.InsertDimOp(data_source_node, insertpos, bcdim)
                    self.graph.addNode(newnode)
                    self.graph.addEdge(query_graph.ParamEdge(data_source_node, newnode,"slice"))
                    data_source_node = newnode
                insertpos = insertpos + 1
                    
        #remove constraint inserts
        cnode = constraint
        while isinstance(cnode, ops.InsertDimOp):
                cnode = self.graph.remove_unaryop(cnode)
        assert cnode == constraint_source_node, "Node mismatch" 
        

        #reconstruct filter op
        newnode = ops.FilterOp(data_source_node, constraint_source_node, d)
        self.graph.addNode(newnode)
        self.graph.addEdge(query_graph.ParamEdge(data_source_node, newnode,"slice"))
        self.graph.addEdge(query_graph.ParamEdge(constraint_source_node, newnode,"constraint"))
        data_source_node = newnode

        #replace bc dims
        ndims = []
        for dim in data_source_node.dims:
            if dim in data_insert_dims_bc:
                bcdim = data_insert_dims[data_insert_dims_bc.index(dim)]
                ndims.append(bcdim)
            else:
                ndims.append(dim)
        data_source_node.dims = dimpaths.DimPath(*ndims)

        #do unpack
        newnode = ops.UnpackArrayOp(data_source_node, len(after) + 1)
        self.graph.addNode(newnode)
        self.graph.addEdge(query_graph.ParamEdge(data_source_node, newnode,"slice"))
        data_source_node = newnode
        

        #redo aggregate
        outparam = repops_funcs.Param(aggnode.name, aggnode.type)
        newnode = ops.UnaryFuncAggregateOp(aggnode.funcname, aggnode.sig, outparam, 
                                            len(after) + 1, data_source_node, **aggnode.kwargs)
        self.graph.addNode(newnode)
        self.graph.addEdge(query_graph.ParamEdge(data_source_node, newnode,"slice"))
        data_source_node = newnode
                                            

        #determine permute_idxs out
        permute_idxs = [data_source_node.dims.index(dim) for dim in aggnode.dims]


        #insert post permute
        newnode = ops.PermuteDimsOp(data_source_node, permute_idxs)
        self.graph.addNode(newnode)
        self.graph.addEdge(query_graph.ParamEdge(data_source_node, newnode,"slice"))
        data_source_node = newnode

        #remove target nodes aggnode
        edges = self.graph.edge_source[aggnode]
        for edge in list(edges):
            self.graph.dropEdge(edge)
            edge.source = data_source_node
            self.graph.addEdge(edge)

        self.graph.pruneGraph()
            

