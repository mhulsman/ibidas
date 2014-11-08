from ..utils import util;
from fasta import write_fasta_text;
import numpy;
import tempfile;
import csv;
import math;
import os;
import fnmatch;
import multiprocessing;
import os.path;
import time
import subprocess
import maf
import argparse
from ibidas import *

  #         0'score',  1'chrom1', 2'pos1',3'length1',4'strand1', 5'chromlength1', 6'chrom2', 7'pos2', 8'length2', 9'strand2', 10'chromlength2',11'mapping'
###############################################################################


def last(data, type, folder=None, pos_mode = 'last', dbargs='', alargs='', lsargs='', probs=0, trf=False, last_split=True, calc_evalue=True,softmask=False,tmpdir=None):
  alargs = [alargs]
  dbargs = [dbargs]
  lsargs = [lsargs]

  if probs:
    alargs.append('-j %d' % (probs+4))

  seq_1 = data[0];
  seq_2 = data[1];

  if type[0] != type[1]:
    if type[0] == 'n' and type[1] == 'p':
        util.warning('Reversing order, last supports only Prot to DNA and not DNA to Prot')
        seq_1,seq_2 = seq_2,seq_1
        type = (type[1],type[0])
  
  if trf:
    seq_1 = [s.upper() for s in seq_1]
    seq_2 = [s.upper() for s in seq_2]
    if not_contains(dbargs, '-c'):
        dbargs.append('-c')
    if not_contains(alargs, '-u'):
        alargs.append('-u2')

  title_1 = [ "%d" % i for i in xrange(len(seq_1)) ];
  title_2 = [ "%d" % i for i in xrange(len(seq_2)) ];

  fas_1 = tempfile.NamedTemporaryFile(suffix='.fasta',dir=tmpdir)
  fas_2 = tempfile.NamedTemporaryFile(suffix='.fasta',dir=tmpdir)
  res = tempfile.NamedTemporaryFile(suffix='.maf',dir=tmpdir)
  
  db_1  = fas_1.name[:-6]
  db_2  = fas_2.name[:-6]

  md5_1 = write_fasta_text(title_1, seq_1, len(seq_1), fas_1);
  md5_2 = write_fasta_text(title_2, seq_2, len(seq_2), fas_2);
  
  fas_1.flush()
  fas_2.flush()
  
  if trf:
    fas_1 = trf_run_CMD(fas_1, seq_1, type[0],softmask=softmask, tmpdir=tmpdir)
    fas_2 = trf_run_CMD(fas_2, seq_2, type[1],softmask=softmask, tmpdir=tmpdir)

  if type[0] != type[1]:
    calc_evalue = False

  util.run_cmd(last_make_db_CMD(db_1, fas_1.name, type[0], dbargs), verbose=False)
  if calc_evalue:
     util.run_cmd(last_make_db_CMD(db_2, fas_2.name, type[1], dbargs), verbose=False)

  util.run_cmd(last_run_CMD(db_1, type[0], db_2, fas_2.name, type[1], alargs, lsargs, last_split, calc_evalue), shell=True, stdout=res, verbose=False)
  res.flush()

  data = last_result2(res.name, pos_mode, probs>0, last_split, calc_evalue); 

  fas_1.close();
  fas_2.close();
  res.close()
  util.run_cmd('rm %s.*' % db_1, shell=True)
  if calc_evalue:
    util.run_cmd('rm %s.*' % db_2, shell=True)

  return data

#edef

def not_contains(args, value):
    return any([value in arg for arg in args])

###############################################################################
def last_result2(resfile, pos_mode = 'last',has_prob=False, last_split=False, calc_evalue=False):
  result = maf.read_maf(resfile)
  #base_last_fields = ('qseqid', 'sseqid', 'qlen','qstart', 'qend', 'qstrand', 'slen', 'sstart', 'send', 'sstrand')
  last_fields = ('name1', 'name2', 'seq_size1', 'start1', _.start1 + _.aln_size1, 'strand1', 'seq_size2', 'start2', _.start2 + _.aln_size2, 'strand2','score','mapping')
  array_types = (int, int, int, int, int, str, int, int, int, str, float, object)
  
  lastprob = 1
  if has_prob:
      last_fields = last_fields + ('prob%d' % lastprob, )
      array_types = array_types + (object,)
      lastprob += 1

  if last_split:
      last_fields = last_fields + ('mismap','prob%d' % lastprob)
      array_types = array_types + (float,object)

  if calc_evalue:
      last_fields = last_fields + ('expect',)
      array_types = array_types + (float,)

  if len(result) == 0:
      cols = tuple([ util.darray([],type) for type in array_types ] )
  else:
      cols = result.Get(*last_fields)()
      cols = tuple([util.darray(col, type) for col,type in zip(cols, array_types)])
  

  if pos_mode == 'last':
      res = cols
  elif pos_mode == 'blast':
      #base_last_fields = (0'qseqid', 1'sseqid', 2'qlen',3'qstart', 4'qend', 5'qstrand', 6'slen', 7'sstart', 8'send', 9'sstrand')
      sstart, send = remove_strand_baseone(cols[7] + 1, cols[8], cols[9], cols[6])
      qstart, qend = remove_strand_baseone(cols[3] + 1, cols[4], cols[5], cols[2])
      f = cols[9] == '-'
      qstart, qend = switch_pos_baseone(qstart, qend, f)
      sstart, send = switch_pos_baseone(sstart, send, f)
      res = (cols[0],cols[1], cols[2], qstart, qend, cols[6], sstart, send) + cols[10:]
  else:
      raise RuntimeError, 'unknown position mode %s' % pos_mode
  return res;


###############################################################################

def switch_strand_baseone(start, end, strand, chromlength, switch_filter=None):
    #switch to other chromosome
    l = end - start + 1
    nstart = chromlength - end + 1
    nend = nstart + l - 1
    if switch_filter is None:
        switch_filter = strand == '-'
    f = ~switch_filter 
    nstart[f] = start[f]
    nend[f] = end[f]
    return (nstart, nend)

def switch_pos_baseone(start, end, switch_filter):
    nstart = start.copy()
    f = switch_filter
    nstart[f] = end[f]
    end[f] = start[f]
    return (nstart, end) 

def remove_strand_baseone(start, end, strand, chromlength):
    f = strand == '-'
    nstart, nend = switch_strand_baseone(start, end, strand, chromlength, f)
    #turn start, end to indicate negative strand
    nnstart = nstart.copy()
    nnstart[f] = nend[f]
    nend[f] = nstart[f]
    return (nnstart, nend)
   

def last_make_db_CMD(dbname, fas_file, type, arguments):
  if type == 'p':
      arguments.append('-p')
  return "lastdb %s %s %s" % (' '.join(arguments), dbname, fas_file);
#edef

###############################################################################

def last_run_CMD(dbname1, type1, dbname2, fas_file2, type2, arguments, lsargs, last_split=True, calc_evalue=True):
  lsargs = ' '.join(lsargs)
  if type1 == 'p' and type2 == 'n':
     arguments.append('-F 15')
  args = ' '.join(arguments)

  base = "lastal %s %s %s" % (args, dbname1, fas_file2)

  if last_split:
        base += ' | last-split %s' % lsargs
  if calc_evalue:
        base += ' | lastex %s.prj %s.prj -' % (dbname1,dbname2)
  #base += ' | maf-convert.py tab'

  return base


def trf_run_CMD(fas_file, seq, seqtype,softmask=False,tmpdir=None):
    if seqtype == 'p':
        return fas_file

    olddir = os.getcwd()
    os.chdir(os.path.dirname(fas_file.name))
    f = open(os.devnull, 'w')
    util.run_cmd("trf %s 2 7 7 80 10 50 2000 -m -h" % fas_file.name, verbose=False,stdout=f)

    fileres = fas_file.name + '.2.7.7.80.10.50.2000.mask'
    fileres2 = fas_file.name + '.2.7.7.80.10.50.2000.dat'
    result = Read(fileres,sep='\t')
    util.run_cmd('rm %s' % fileres, shell=True)
    util.run_cmd('rm %s' % fileres2, shell=True)
    fas_file.close()
     
    fas_new = tempfile.NamedTemporaryFile(suffix='.fasta', tmpdir=tmpdir)
    if softmask:
        nresult = []
        for seqold,seqnew in zip(seq, result.seq()):
            seqnew = ''.join([c1 if c2 != 'N' or c1 == 'N' else c1.lower() for c1,c2 in zip(seqold, seqnew)])
            nresult.append(seqnew)
        write_fasta_text(result.Get(0)(), nresult, len(nresult), fas_new)
    else:
        write_fasta_text(result.Get(0)(), result.seq(), len(result), fas_new)
    fas_new.flush()

    os.chdir(olddir)
    return fas_new


def parsemapping(mapping):#{{{
    #parses LAST tab mapping format to list of integers and tuples of integers. 
    mq = []
    mapping = mapping.split(',')
    for m in mapping:
        if ':' in m:
            f,t = m.split(':')
            mq.append((int(f), int(t)))
        else:
            mq.append(int(m))
    return mq#}}}

