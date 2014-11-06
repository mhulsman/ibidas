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
###############################################################################


def last(data, type, folder=None, pos_mode = 'last', dbargs='-c -uMAM8', alargs='-x50 -m100', lsargs='', trf=False, last_split=True, calc_evalue=True):
  alargs = [alargs]
  dbargs = [dbargs]
  lsargs = [lsargs]

  seq_1 = data[0];
  seq_2 = data[1];

  if type[0] != type[1]:
    if type[0] == 'n' and type[1] == 'p':
        util.warning('Reversing order, last supports only Prot to DNA and not DNA to Prot')
        seq_1,seq_2 = seq_2,seq_1
        type = (type[1],type[0])
  
  if trf:
    seq_1 = seq_1.upper()
    seq_2 = seq_2.upper()
    if not_contains(dbargs, '-c'):
        dbargs.append('-c')
    if not_contains(alargs, '-u'):
        alargs.append('-u2')

  title_1 = [ "%d" % i for i in xrange(len(seq_1)) ];
  title_2 = [ "%d" % i for i in xrange(len(seq_2)) ];

  fas_1 = tempfile.NamedTemporaryFile(suffix='.fasta')
  fas_2 = tempfile.NamedTemporaryFile(suffix='.fasta')
  res = tempfile.NamedTemporaryFile(suffix='.tsv')
  
  db_1  = fas_1.name[:-6]
  db_2  = fas_2.name[:-6]

  md5_1 = write_fasta_text(title_1, seq_1, len(seq_1), fas_1);
  md5_2 = write_fasta_text(title_2, seq_2, len(seq_2), fas_2);
  
  fas_1.flush()
  fas_2.flush()
  
  if trf:
    fas_1 = run_trf(fas_1, type[0])
    fas_2 = run_trf(fas_2, type[2])

  if type[0] != type[1]:
    calc_evalue = False

  util.run_cmd(last_make_db_CMD(db_1, fas_1.name, type[0], dbargs), verbose=False)
  if calc_evalue:
     util.run_cmd(last_make_db_CMD(db_2, fas_2.name, type[1], dbargs), verbose=False)

  util.run_cmd(last_run_CMD(db_1, type[0], db_2, fas_2.name, type[1], alargs, lsargs, last_split, calc_evalue), shell=True, stdout=res, verbose=False)
  res.flush()

  data = last_result(res.name, pos_mode); 

  fas_1.close();
  fas_2.close();
  res.close()
  util.run_cmd('rm %s.*' % db_1, shell=True)
  if calc_evalue:
    util.run_cmd('rm %s.*' % db_2, shell=True)

  return data

#edef

def not_contains(args):
    return any([value in arg for arg in args])

###############################################################################

def last_result(resfile, pos_mode = 'last'):
  bm = {};
  br = open(resfile, 'rb');

  #              'score',  'chrom1', 'pos1','length1','strand1', 'chromlength1', 'chrom2', 'pos2', 'length2', 'strand2', 'chromlength2','mapping'
  sp_types = (float,    int,          int,   int,      str,        int,            int,      int,    int,       str,      int,          parsemapping)

  rdr = csv.reader(br, delimiter='\t', quotechar='"');
  rows = []
  for row in rdr:
    if row[0].startswith('#'):
        continue
    row = [ sp_type(elem) for sp_type,elem in zip(sp_types, row) ];
    rows.append(row)
  
  #              'score',  'chrom1', 'pos1','length1','strand1', 'chromlength1', 'chrom2', 'pos2', 'length2', 'strand2', 'chromlength2','mapping'
  array_types = (float,    int,      int,   int,      str,        int,            int,      int,    int,       str,      int,     object)
 
  if len(rows) > 0:
    cols = util.transpose_table(rows)
    cols = tuple([util.darray(col, type) for col,type in zip(cols, array_types)])
  else:
    cols = tuple([ util.darray([],type) for type in array_types ] )
  
  #         qseqid     sseqid   qlen      qstart   qend               slen sstart send length mismatch gapopen pident evalue bitscore
  #         0'score',  1'chrom1', 2'pos1',3'length1',4'strand1', 5'chromlength1', 6'chrom2', 7'pos2', 8'length2', 9'strand2', 10'chromlength2',11'mapping'

  if pos_mode == 'last':
        res = (cols[1],      cols[6], cols[5], cols[2], cols[2] + cols[3], cols[4], cols[10], cols[7], cols[7] + cols[8], cols[9],cols[0], cols[11])
  elif pos_mode == 'blast':
        sstart, send = remove_strand_baseone(cols[7] + 1, cols[7] + cols[8], cols[9], cols[10])
        qstart, qend = remove_strand_baseone(cols[2] + 1, cols[2] + cols[3], cols[4], cols[5])
        f = cols[9] == '-'
        qstart, qend = switch_pos_baseone(qstart, qend, f)
        sstart, send = switch_pos_baseone(sstart, send, f)
        res = (cols[1],cols[6], cols[5], qstart, qend, cols[10], sstart, send, cols[0], cols[11])
  else:
    raise RuntimeError, 'unknown positioni mode %s' % pos_mode
  br.close();
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

def run_trf(fas_file, type):
    if type == 'p':
        return

    return fas_file

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
  base += ' | maf-convert.py tab'

  return base


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

