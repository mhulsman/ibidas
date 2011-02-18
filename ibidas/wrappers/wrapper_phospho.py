import wrapper_tsv
_delay_import_(globals(),"wrapper_py","rep")

def read_phospho(path):
    fpath = path + 'kinase_all.txt' 
    kinase_list = wrapper_tsv.read_tsv(fpath)
    kinase_list = kinase_list[kinase_list.map(lambda x:x[0] != "#", otype="bool")]

    data = []
    dtype = "[kinase_binding:*](tuple(kinase=string[], orf=string[], score=real64))"
    for elem in kinase_list():
        filename = path + str(elem) + ".txt"
        file = open(filename)
        file.next()
        for row in file:
            r = row.split('\t')
            data.append((str(elem), r[0], float(r[1])))

    return Rep(data,dtype)
