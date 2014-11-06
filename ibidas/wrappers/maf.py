from python import Rep;
from ..utils import util
import numpy

seqconv = [str, int, int, str, int, str]
def read_maf(fname, types=()):
    f = util.open_file(fname, mode='rU')
    
    records = []
    cur_rec = {}

    for line in f:
        line = line.strip()
        if len(line) == 0:
            if cur_rec:
                records.append(process_finished_record(cur_rec, types))
                cur_rec = {}
            continue

        line_type = line[0]
        line = line[2:]
        if line_type == '#':
            continue
        elif line_type == 'a':
            elements = line.split(' ')
            arguments = [elem.split('=') for elem in elements if elem]
            arguments = [(util.valid_name(key), value) for key,value in arguments]
            cur_rec.update(dict(arguments))
        elif line_type == 's':
            i = 1
            while ('name%d' % i) in cur_rec: i += 1
            elements = line.split(' ')
            data = [elem for elem in elements if elem]
            assert len(data) == 6, 'Number of data elements on a seq line is expected to be 6'
            data = [conv(elem) for conv,elem in zip(seqconv,data)]
            cur_rec['name%d' % i], cur_rec['start%d' % i], cur_rec['aln_size%d' % i], cur_rec['strand%d' % i], cur_rec['seq_size%d' %i], cur_rec['alignment%d' % i] = data
        elif line_type == 'p':
            i = 1
            while ('prob%d' % i) in cur_rec: i += 1
            data = line.strip()
            probs = numpy.array([10**(-((ord(char) - 33) / 10.)) for char in data],dtype=float)
            cur_rec['prob%d' % i] = probs
        else:
            pass

    if cur_rec:
        records.append(process_finished_record(cur_rec, types))

    return Rep(records).Copy()


def process_finished_record(rec, types):
    if 'alignment1' in rec and 'alignment2' in rec and not 'alignment3' in rec:
        rec['mapping'] = alignment_to_mapping(rec.pop('alignment1'), rec.pop('alignment2', *types))
    return rec

def alignment_to_mapping(s1, s2,type1='n',type2='n'):#{{{
    addvalue1 = 1
    addvalue2 = 1
    
    if type1 == 'n' and type2=='p': addvalue1 = 3
    elif type1 == 'p' and type2=='n': addvalue2 = 3

    mapping = []
    cur_value = None 
    for c1, c2 in zip(s1,s2):
        x1 = c1 != '-'
        x2 = c2 != '-'

        if x1 and x2:
            if not isinstance(cur_value,int):
                mapping.append(cur_value)
                cur_value = 0
            cur_value += 1
        elif x1:
            if not isinstance(cur_value, list) or cur_value[0] == 0:
                mapping.append(cur_value)
                cur_value = [0,0]
            r1a = c1 == "\\"
            r1b = c1 == "/"
            if r1a:
                cur_value[0] += 1 
            elif r1b:
                cur_value[0] -= 1
            else:
                cur_value[0] += addvalue1
        elif x2:
            if not isinstance(cur_value, list) or cur_value[1] == 0:
                mapping.append(cur_value)
                cur_value = [0,0]
            r2a = c2 == "\\"
            r2b = c2 == "/"
            if r2a:
                cur_value[1] += 1 
            elif r2b:
                cur_value[1] -= 1
            else:
                cur_value[1] += addvalue2
        
        else:
            raise RuntimeError, 'Double gap in %s and %s' % (s1,s2)
    mapping.append(cur_value)
    mapping = mapping[1:] #remove none
    pos = 0
    while pos < (len(mapping) - 1):
        if isinstance(mapping[pos],list) and isinstance(mapping[pos+1], list):
            x = mapping[pos]
            x[0] += mapping[pos+1][0]
            x[1] += mapping[pos+1][1]
            del mapping[pos+1]
        else:            
            pos += 1

    mapping = list([tuple(m) if isinstance(m,list) else m for m in mapping])
    return mapping#}}}



