from .. import Rep;
from .. utils import util;

################################################################################

def read_fasta(fname, sep='|', split=1, fieldnames=()):

  f = util.open_file(fname,mode='rU');
  fas = [];

  seqid = "";
  seq   = "";

  for line in f:
    line = line.strip();
    if not line or line[0] == ">":
      if seqid:
        fas = fas + prep_seq_tuple(seqid, seq, sep=sep, split=split);
      #fi
      seqid = line[1:];
      seq = "";
      continue;
    #fi
    seq = seq + line;
  #efor

  f.close();

    # check for one last sequence
  fas = fas + prep_seq_tuple(seqid, seq, sep=sep, split=split) if seq else fas;

  return Rep(fas)/fieldnames;

#edef

################################################################################

def prep_seq_tuple(seqid, seq, sep='|', split=1):
  seqid = tuple(x.strip() for x in seqid.split(sep)) if split else ( seqid, );
  return [ seqid + ( seq, ) ];
#edef

################################################################################

