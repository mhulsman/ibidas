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


def last(data, type, folder=None, pos_mode = 'last', dbargs='-c -uMAM8', alargs='-e10 -m100', lpargs='', last_split=True):

  seq_1 = data[0];
  seq_2 = data[1];

  title_1 = [ "%d" % i for i in xrange(len(seq_1)) ];
  title_2 = [ "%d" % i for i in xrange(len(seq_2)) ];

  fas_1 = tempfile.NamedTemporaryFile(suffix='.fasta')
  fas_2 = tempfile.NamedTemporaryFile(suffix='.fasta')
  res = tempfile.NamedTemporaryFile(suffix='.tsv')
  db_2  = fas_2.name[:-6]

  md5_1 = write_fasta_text(title_1, seq_1, len(seq_1), fas_1);
  md5_2 = write_fasta_text(title_2, seq_2, len(seq_2), fas_2);
  fas_1.flush()
  fas_2.flush()

  util.run_cmd(last_make_db_CMD(db_2, fas_2.name, type, dbargs), verbose=True)
  util.run_cmd(last_run_CMD(db_2, fas_1.name, alargs, lpargs, last_split), shell=True, stdout=res, verbose=True)
  res.flush()
  data = last_result(db_2, fas_1.name, res.name, pos_mode); # if reciprocal blast_reciprocal(file_12.name, file_21.name) else blast_res_to_dict(file_12.name)

  fas_1.close();
  fas_2.close();
  res.close()
  util.run_cmd('rm %s.*' % db_2, shell=True)

  return data

#edef

###############################################################################

def last_result(dbname, fas_file, resfile, pos_mode = 'last'):
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
        res = (cols[6],      cols[1], cols[10], cols[7], cols[7] + cols[8], cols[9],cols[5], cols[2], cols[2] + cols[3], cols[4], cols[0], cols[11])
  elif pos_mode == 'blast':
        qstart, qend = remove_strand_baseone(cols[7] + 1, cols[7] + cols[8], cols[9], cols[10])
        sstart, send = remove_strand_baseone(cols[2] + 1, cols[2] + cols[3], cols[4], cols[5])
        f = cols[9] == '-'
        qstart, qend = switch_pos_baseone(qstart, qend, f)
        sstart, send = switch_pos_baseone(sstart, send, f)
        res = (cols[6],cols[1], cols[10], qstart, qend, cols[5], sstart, send, cols[0], cols[11])
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
    nstart[f] = start
    nend[f] = end
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
   

def last_make_db_CMD(dbname, fas_file, type, args=''):
  arguments = []
  if type == 'prot':
      arguments.append('-p')
  arguments.append(args)

  return "lastdb %s %s %s" % (' '.join(arguments), dbname, fas_file);
#edef

###############################################################################

def last_run_CMD(dbname, fas_file, args='', lpargs='', last_split=True):
  if last_split:
      return "lastal %s %s %s | last-split %s | maf-convert.py tab" % (args, dbname, fas_file, lpargs);
  else:
      return "lastal %s %s %s | maf-convert.py tab" % (args, dbname, fas_file);
#edef

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

