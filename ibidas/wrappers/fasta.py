from .. import Rep;
from .. utils import util;

################################################################################

def read_fasta(fname, sep='auto', split=1, fieldnames=()):

    

    f = util.open_file(fname,mode='rU');
    fas = [];

    seqid = "";
    seq   = "";

    for line in f:
        line = line.strip('\n\r');
        if not line or line[0] == ">":
            if seqid:
                fas.append((seqid, seq))
            seqid = line[1:];
            seq = "";
            continue;
        seq = seq + line;

    f.close();

      # check for one last sequence
    if seq:
        fas.append((seqid,seq))

    if split:
        if sep == 'auto':
            sep_search = ['\t','|',',']
            for sep in sep_search:
                if all([sep in fas[i][0] for i in xrange(min(len(fas),100))]):
                    break
            else:
                util.warning('Could not determine FASTA separator. Please specify through sep parameter')
                sep = ''

        if sep:
            fas = [tuple(x.strip() for x in row[0].split(sep)) + row[1:] for row in fas]
            
    return Rep(fas)/fieldnames;


################################################################################


