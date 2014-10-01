from python import Rep;
from ..utils import util;

################################################################################

def read_fasta(fname, sep='auto', fieldnames=()):

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
                seqs.append(seq.replace(' ',''));
            seqid = line[1:];
            seq = "";
            continue;
        seq = seq + line;

    f.close();

      # check for one last sequence
    if seq:
        fas.append(seqid);
        seqs.append(seq.replace(' ',''));

    maxf = 0;
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
        maxf = 1

    if not fieldnames:
        fieldnames = tuple(['f%d' %i for i in range(maxf)] + ['seq'])
            
    return Rep(fas)/fieldnames;


################################################################################

def write_fasta_text(title, seq, nseq, fout, sep=70):

  import md5;
  
  m = md5.new();
  
  for i in xrange(nseq):
    s = seq[i];
    line = ">%s\n" % (title[i]);
    fout.write(line);
    m.update(line);
    for i in range(0, len(s), sep):
      line = "%s\n" % (s[i:i+sep]);
      fout.write(line);
      m.update(line);
    #efor
  #efor
  return m.hexdigest();

##############################################################################

def write_fasta(data, filename, **kwargs):
  save_fasta_rep(data, open(filename, 'w'), **kwargs);
#edef

##############################################################################

def save_fasta_rep(data, fout, **kwargs):

  from ibidas.itypes import rtypes;

  nseq = data.Shape()();
  
  if nseq == 0:
    print "Empty FASTA file";
    return 1;
  #fi

  vs  = kwargs.pop('vs', '|');
  sep = kwargs.pop('sep', 70);
  seqslice = kwargs.pop('seqslice', None);
  
  names = data.Names;
  if seqslice is None:
    seqslice = [ i for i in xrange(len(data._slices)) if isinstance(data._slices[i].type, rtypes.TypeSequence) ];
  else:
    seqslice = set(data.Get(seqslice)._slices)
    seqslice = [ i for i in xrange(len(data._slices)) if data._slices[i] in seqslice ];
  #fi

  if len(seqslice) > 1:
    seqslice = seqslice[-1];
    util.warning("More than one sequence slice, using slice '%s'." % data._slices[seqslice].name);
  if len(seqslice) == 0:
    seqslice = len(names) - 1;
    util.warning("No sequence slice specified, using last slice: '%s'." % data._slices[seqslice].name);
  else:
    seqslice = seqslice[-1];
  #fi
  
  data  = data.Cast(*["bytes" for x in data.Names])();
  title = data[0:seqslice] + data[(seqslice+1):len(data)];
  title = [ vs.join(x) for x in zip(*title)]
  seq   = data[seqslice];
  
  return write_fasta_text(title, seq, nseq, fout, sep=sep);

#edef

##############################################################################

def read_fastq(fname, **kwargs):
  print fname
  fd = util.open_file(fname, mode='rU');
  
  seqs = [];
  
  while True:
        seqid = fd.readline();

        if seqid == '':
            break;
        #fi

        if seqid[0] == '@':
            seq = fd.readline();
            fd.readline() # Remove '+'
            qlty = fd.readline();
            seqs = seqs + [ (seqid[0:-1], seq[0:-1], qlty[0:-1]) ];
        #fi
  #ewhile
  fd.close()
  
  return Rep(seqs) / ('sequenceid', 'sequence', 'quality');
#edef

