from .. import Rep;
from .. utils import util;

################################################################################

def read_fasta(fname, sep='auto', split=1, fieldnames=()):

    

    f = util.open_file(fname,mode='rU');
    fas  = [];
    seqs = []

    seqid = "";
    seq   = "";

    for line in f:
        line = line.strip('\n\r');
        if not line or line[0] == ">":
            if seqid:
                fas.append(seqid);
		seqs.append(seq);
            seqid = line[1:];
            seq = "";
            continue;
        seq = seq + line;

    f.close();

      # check for one last sequence
    if seq:
        fas.append(seqid);
	seqs.append(seq);

    maxf = 0;
    if split:
        if sep == 'auto':
            sep_search = ['\t','|',', ',',']
            for sep in sep_search:
                if all([sep in fas[i][0] for i in xrange(min(len(fas),100))]):
                    break
            else:
                util.warning('Could not determine FASTA separator. Please specify through sep parameter')
                sep = ''

        if sep:
	    fsid = [ tuple(x.strip() for x in util.resplit(row, sep, "\"'")) for row in fas ];
	    maxf = max([ len(x) for x in fsid]);
	    fas = [ tuple([ fsid[i][j] if j < len(fsid[i]) else '' for j in xrange(maxf)]) + (seqs[i],) for i in xrange(len(fas)) ]
	else:
	    fas = [ (fas[i], seqs[i]) for i in xrange(len(fas)) ];
            
    return Rep(fas)/fieldnames;


################################################################################


