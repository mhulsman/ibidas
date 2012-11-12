from .. import Rep;

def read_fasta(fname, sep='|', split=1, fieldnames=()):

  f = open(fname, "r");
  fas = [];

  seqid = "";
  seq   = "";

  for line in f:
    line = line.strip();
    if not line or line[0] == ">":
      if seqid:
	seqid = tuple(x.strip() for x in seqid.split(sep)) if split else ( seqid, );
        fas = fas + [ seqid + ( seq, ) ];
      #fi
      seqid = line[1:];
      seq = "";
      continue;
    #fi
    seq = seq + line;
  #efor

  f.close();

  return Rep(fas)/fieldnames;

#edef

