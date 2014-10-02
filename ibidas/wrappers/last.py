from ..utils import util;
from fasta import write_fasta_text;
import numpy;
import tempfile;
import csv;
import math;
import os;
import fnmatch;
import multiprocessing;

###############################################################################


def last(data, type, folder=None, dbargs='-c -uMAM8', alargs='-e10 -m100'):

  seq_1 = data[0];
  seq_2 = data[1];

  title_1 = [ "%d" % i for i in xrange(len(seq_1)) ];
  title_2 = [ "%d" % i for i in xrange(len(seq_2)) ];

  fas_1 = tempfile.NamedTemporaryFile(suffix='.fasta')
  fas_2 = tempfile.NamedTemporaryFile(suffix='.fasta')
  db_2  = fas_2.name[:-6]

  md5_1 = write_fasta_text(title_1, seq_1, len(seq_1), fas_1);
  md5_2 = write_fasta_text(title_2, seq_2, len(seq_2), fas_2);

  CMDs = [ last_make_db_CMD(db_2, fas_2.name, type, dbargs) ] + [ last_run_CMD(db2, fas_1.name, alargs) ];

  util.run_seq_cmds(CMDs);
  
  data = last_result(db2, fas1.name); # if reciprocal blast_reciprocal(file_12.name, file_21.name) else blast_res_to_dict(file_12.name)

  fas_1.close();
  fas_2.close();
  resfile.close()

  return data

#edef

###############################################################################

def last_result(dbname, fas_file):
  bm = {};
  br = open('%s_%s.tsv' % (dbname, fas_file), 'rb');

  #              'score',  'chrom1', 'pos1','length1','strand1', 'chromlength1', 'chrom2', 'pos2', 'length2', 'strand2', 'chromlength2','mapping'
  array_types = (float,    str,      int,   int,      str,        int,            str,      int,    int,       str,      int,     parsemapping)

  rdr = csv.reader(br, delimiter='\t', quotechar='"');
  rows = []
  for row in rdr:
    if row[0].startswith('#'):
        continue
    row = [ sp_types[i](row[i]) for i in xrange(len(row)) ];
    rows.append(row)
  
  #              'score',  'chrom1', 'pos1','length1','strand1', 'chromlength1', 'chrom2', 'pos2', 'length2', 'strand2', 'chromlength2','mapping'
  array_types = (float,    str,      int,   int,      str,        int,            str,      int,    int,       str,      int,     object)
 
  if len(rows) > 0:
    cols = util.transpose_table(rows)
    cols = tuple([util.darray(col, type) for col,type in zip(cols, array_types)])
  else:
    cols = tuple([ util.darray([],type) for type in array_types ] )
    
  #         qseqid  sseqid qlen qstart qend slen sstart send length mismatch gapopen pident evalue bitscore

  #ols[1], cols[6], cols[5], cols[2], 
  br.close();
  return cols;

###############################################################################

def last_make_db_CMD(dbname, fas_file, type, args='-c -uMAM8'):
  #makeblastdb -in "fas_file" -out "filename" -dbtype "prot"
  arguments = []
  if type == 'prot':
      arguments.append('-p')
  arguments.append(args)

  return "lastdb %s %s %s" % (dbname, fas_file, ' '.join(arguments));
#edef

###############################################################################

def last_run_CMD(dbname, fas_file, args='-e10 -m100'):
  return "lastal %s %s %s | last-split | maf-convert tab > %s_%s.tsv" % (args, dbname, fas_file, dbname, fas_file);
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

