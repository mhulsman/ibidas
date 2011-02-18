import os

import wrapper
_delay_import_(globals(),"wrapper_py","Result","rep")
_delay_import_(globals(),"slices")
_delay_import_(globals(),"..itypes","rtypes")


def read_chipchip(path):
    files = os.listdir(path)
    pval_names = ['none', 'p0.005', 'p0.001']
    max_pvals = [1.0, 0.005, 0.001]
    cons = [0, 1, 2]

    loaded_files = []
    for mpval, pval_name in zip(max_pvals, pval_names):
        for con in cons:
            filename = path + "orfs_by_factor_%s_cons%s.txt" % (pval_name, str(con))
            res = ChipChipRepresentor(filename)
            res = res.E.A.join(Rep([(mpval, con)])/("max_pval", "conservation"))
            res = res.flat().tuple().array()
            loaded_files.append(res)

    res = sum(loaded_files[1:], loaded_files[0]).E.A
    res = res.group_by(_.regulator, name="orfs")
    res = res.get(_.regulator, _.Dorfs.group_by(_.orfs))
    res = res.get(_.regulator, _.orfs, _.max_pval.min(), _.conservation.max())

    return res.copy()
    

cctype = "[regulators:*](tuple(regulator=string[], orfs=[orfs:~](string[])))"

class ChipChipRepresentor(wrapper.SourceRepresentor):
    _select_indicator = None
    _select_executor = None
    
    def __init__(self, filename):
        self._filename = filename 
        
        ntype = rtypes.createType(cctype)
        tslices = (slices.Slice("data", ntype),)

        all_slices = dict([(slice.id, slice) for slice in tslices])
        wrapper.SourceRepresentor.__init__(self, all_slices, tslices)

    def pyexec(self, executor):
        results = []
        file = open(self._filename)
        for row in file:
            res = row.split('\t')
            results.append((res[0], res[1:-1]))

        res = {}
        res[self._active_slices[0].id] = results
        return Result(res)


