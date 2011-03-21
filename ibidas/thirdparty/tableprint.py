#from http://code.activestate.com/recipes/267662-table-indentation/

import cStringIO,operator
import numpy
from ..utils import util

def calc_width(rows):
    width = numpy.zeros((len(rows),len(rows[0])),dtype=int)
    for pos,row in enumerate(rows):
        width[pos,:] = map(len,row)
    return width.max(axis=0)
    
def advise_splits(console_width, rows_width, border_width, mincol_width=40):
    nrows_width = rows_width.copy()

    nrows_width[rows_width > mincol_width] = 40

    indices = perform_split(nrows_width, border_width, console_width)
    return indices
    
def optimize_width(console_width, rows_width, border_width):
    nrows_width = rows_width.copy()
    while (nrows_width.sum() + border_width * (len(nrows_width) - 1)) > console_width:
        maxval = nrows_width.max()
        nmax = (nrows_width == maxval).sum()
        rem_lengths = nrows_width[nrows_width != nrows_width.max()]
        tomuch = nrows_width.sum() + border_width * (len(nrows_width) - 1) - console_width
        perrow = numpy.ceil(tomuch / float(nmax))

        if(len(rem_lengths) == 0):
            nrows_width -= perrow
        else:
            next_maxval = rem_lengths.max()
            nrows_width[nrows_width == maxval] -= min(perrow, maxval - next_maxval)
    return nrows_width

def perform_split(rows_width, border_width, consolewidth):
    indices = []
    start = 0
    while(start < len(rows_width)):
        stop = start + 1
        while(stop <= len(rows_width) and not (rows_width[start:stop].sum() + border_width * (stop - start - 1)) > consolewidth):
            stop += 1
        indices.append(slice(start,stop))
        start = stop

    

    return indices
    
def row_crop(row_widths, rows):
    nrows = []
    for row in rows:
        nrows.append([crop(col, cwidth) for col, cwidth in zip(row, row_widths)])
    return nrows

def select_cols(rows, indices):
    nrows = []
    for row in rows:
        nrows.append(row[indices])
    return nrows

def prepend_col(rows, col):
    nrows = []
    for elem, row in zip(col, rows):
        nrows.append([elem] + list(row))
    return nrows
    

def indent(rows, hasHeader=False, headerChar='-', delim=' | ', justify='left',
           separateRows=False, prefix='', postfix='', wrapfunc=lambda x:x):
    """Indents a table by column.
       - rows: A sequence of sequences of items, one sequence per row.
       - hasHeader: True if the first row consists of the columns' names.
       - headerChar: Character to be used for the row separator line
         (if hasHeader==True or separateRows==True).
       - delim: The column delimiter.
       - justify: Determines how are data justified in their column. 
         Valid values are 'left','right' and 'center'.
       - separateRows: True if rows are to be separated by a line
         of 'headerChar's.
       - prefix: A string prepended to each printed row.
       - postfix: A string appended to each printed row.
       - wrapfunc: A function f(text) for wrapping text; each element in
         the table is first wrapped by this function."""
    # closure for breaking logical rows to physical, using wrapfunc
    def rowWrapper(row):
        newRows = [wrapfunc(item).split('\n') for item in row]
        if len(row) > 1:
            newRows = map(None, *newRows)
            
        return [[substr or '' for substr in item] for item in newRows]
    # break each logical row into one or more physical ones
    logicalRows = [rowWrapper(row) for row in rows]
    # columns of physical rows
    columns = map(None,*reduce(operator.add,logicalRows))
    # get the maximum of each column by the string length of its items
    maxWidths = [max([len(str(item)) for item in column]) for column in columns]
    rowSeparator = headerChar * (len(prefix) + len(postfix) + sum(maxWidths) + \
                                 len(delim)*(len(maxWidths)-1))
    # select the appropriate justify method
    justify = {'center':str.center, 'right':str.rjust, 'left':str.ljust}[justify.lower()]
    output=cStringIO.StringIO()
    if separateRows: print >> output, rowSeparator
    for physicalRows in logicalRows:
        for row in physicalRows:
            print >> output, \
                prefix \
                + delim.join([justify(str(item),width) for (item,width) in zip(row,maxWidths)]) \
                + postfix
        if separateRows or hasHeader: print >> output, rowSeparator; hasHeader=False
    return output.getvalue()

# written by Mike Brown
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/148061
def wrap_onspace(text, width):
    """
    A word-wrap function that preserves existing line breaks
    and most spaces in the text. Expects that existing line
    breaks are posix newlines (\n).
    """
    return reduce(lambda line, word, width=width: '%s%s%s' %
                  (line,
                   ' \n'[(len(line[line.rfind('\n')+1:])
                         + len(word.split('\n',1)[0]
                              ) >= width)],
                   word),
                  text.split(' ')
                 )

import re
def wrap_onspace_strict(text, width):
    """Similar to wrap_onspace, but enforces the width constraint:
       words longer than width are split."""
    wordRegex = re.compile(r'\S{'+str(width)+r',}')
    return wrap_onspace(wordRegex.sub(lambda m: wrap_always(m.group(),width),text),width)

import math
def wrap_always(text, width):
    """A simple word-wrap function that wraps text on exactly width characters.
       It doesn't split the text in words."""
    return '\n'.join([ text[width*i:width*(i+1)] \
                       for i in xrange(int(math.ceil(1.*len(text)/width))) ])

def crop(text, width):
    if(len(text) > width):
        if(width > 1):
            return text[:(width - 1)] + "~"
        else:
            return text[:width]
    return text

if __name__ == '__main__':
    labels = ('First Name', 'Last Name', 'Age', 'Position')
    data = \
    '''John,Smith,24,Software Engineer
       Mary,Brohowski,23,Sales Manager
       Aristidis,Papageorgopoulos,28,Senior Reseacher'''
    rows = [row.strip().split(',')  for row in data.splitlines()]

    print 'Without wrapping function\n'
    print indent([labels]+rows, hasHeader=True)
    # test indent with different wrapping functions
    width = 10
    for wrapper in (wrap_always,wrap_onspace,wrap_onspace_strict):
        print 'Wrapping function: %s(x,width=%d)\n' % (wrapper.__name__,width)
        print indent([labels]+rows, hasHeader=True, separateRows=True,
                     prefix='| ', postfix=' |',
                     wrapfunc=lambda x: wrapper(x,width))
    
    # output:
    #
    #Without wrapping function
    #
    #First Name | Last Name        | Age | Position         
    #-------------------------------------------------------
    #John       | Smith            | 24  | Software Engineer
    #Mary       | Brohowski        | 23  | Sales Manager    
    #Aristidis  | Papageorgopoulos | 28  | Senior Reseacher 
    #
    #Wrapping function: wrap_always(x,width=10)
    #
    #----------------------------------------------
    #| First Name | Last Name  | Age | Position   |
    #----------------------------------------------
    #| John       | Smith      | 24  | Software E |
    #|            |            |     | ngineer    |
    #----------------------------------------------
    #| Mary       | Brohowski  | 23  | Sales Mana |
    #|            |            |     | ger        |
    #----------------------------------------------
    #| Aristidis  | Papageorgo | 28  | Senior Res |
    #|            | poulos     |     | eacher     |
    #----------------------------------------------
    #
    #Wrapping function: wrap_onspace(x,width=10)
    #
    #---------------------------------------------------
    #| First Name | Last Name        | Age | Position  |
    #---------------------------------------------------
    #| John       | Smith            | 24  | Software  |
    #|            |                  |     | Engineer  |
    #---------------------------------------------------
    #| Mary       | Brohowski        | 23  | Sales     |
    #|            |                  |     | Manager   |
    #---------------------------------------------------
    #| Aristidis  | Papageorgopoulos | 28  | Senior    |
    #|            |                  |     | Reseacher |
    #---------------------------------------------------
    #
    #Wrapping function: wrap_onspace_strict(x,width=10)
    #
    #---------------------------------------------
    #| First Name | Last Name  | Age | Position  |
    #---------------------------------------------
    #| John       | Smith      | 24  | Software  |
    #|            |            |     | Engineer  |
    #---------------------------------------------
    #| Mary       | Brohowski  | 23  | Sales     |
    #|            |            |     | Manager   |
    #---------------------------------------------
    #| Aristidis  | Papageorgo | 28  | Senior    |
    #|            | poulos     |     | Reseacher |
    #---------------------------------------------
