import os
import csv

import wrapper
_delay_import_(globals(),"..itypes","rtypes")
_delay_import_(globals(),"..slices")
_delay_import_(globals(),"wrapper_py","Result","rep")
_delay_improt_(globals(),"wrapper_tsv","read_tsv")
_delay_import_(globals(),"..utils","cutils","util")

def read_hughes(path):
    experiments = read_tsv(path + "/experiment_list.txt", skiprows=2)
    return (experiments, HughesRepresentor(path + "/data_expts1-300_geneerr.txt").A, 
                         HughesRepresentor(path + "/control_expts1-63_geneerr.txt").A)

    

htype = "tuple(gene_official=[genes:*](string[]), gene_symbols=[genes:*](string[]), " + \
       "knockouts=[knockouts:*](string[]), pval=[genes:*]([knockouts:*](real64$)), " + \
               "ratio=[genes:*]([knockouts:*](real64$)), intensity=[genes:*]([knockouts:*](real64$)), " + \
               "ratioerr=[genes:*]([knockouts:*](real64$)))"

class HughesRepresentor(wrapper.SourceRepresentor):
    _select_indicator = None
    _select_executor = None
    
    def __init__(self, filename):
        self._filename = filename
        
        ntype = rtypes.createType(htype)
        tslices = (slices.Slice("data", ntype),)

        all_slices = dict([(slice.id, slice) for slice in tslices])
        wrapper.SourceRepresentor.__init__(self, all_slices, tslices)

    def pyexec(self, executor):
        file = open(self._filename)

        #skip first two lines
        file.readline()
        file.readline()


        reader = csv.reader(file, delimiter='\t')
        knockout_names = reader.next()[4::4]
        meas_names = reader.next()[2:]
        data = [row for row in reader]

        data = util.darray(data,object,2)
        gene_names = data[:, 0]
        gene_symbols = data[:,1]
        intensity = data[:, 2:-1:4]
        ratio = data[:, 3::4]
        ratio_err = data[:, 4::4]
        pval = data[:, 5::4]

        results = (gene_names, gene_symbols, knockout_names, pval, ratio, intensity, ratio_err)
        res = {}
        res[self._active_slices[0].id] = results
        return Result(res)


