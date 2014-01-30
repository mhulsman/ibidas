import sys
import getopt
import os

req_version = (2,6)
cur_version = sys.version_info

if not ((cur_version[0] > req_version[0]) or (cur_version[0] == req_version[0] and cur_version[1] >= req_version[1])):
    sys.stderr.write("Your python interpreter is too old. Ibidas needs at least Python 2.6. Please consider upgrading.\n")
    sys.exit(-1)

#make sure we import the development version
try:
    sys.path = [os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..')), '.'] + sys.path
except:
    pass

from ibidas import *
from ibidas.server import Serve
from ibidas import representor

import IPython

ipversion = IPython.__version__.split('.')
oldip = int(ipversion[0]) == 0 and int(ipversion[1]) <= 10
oldip2 = int(ipversion[0]) == 1 and int(ipversion[1]) < 1

if oldip:
    from IPython.Shell import IPShellEmbed
    from IPython.ipapi import TryNext
    from IPython.genutils import dir2
    import IPython.rlineimpl as readline
elif oldip2:
    from IPython.config.loader import Config
    from IPython.frontend.terminal.embed import InteractiveShellEmbed
else:   
    from IPython.config.loader import Config
    from IPython.terminal.embed import InteractiveShellEmbed
    
    #from IPython.ipapi import TryNext
    #from IPython.genutils import dir2
    #import IPython.rlineimpl as readline

from ibidas.command_parser import parseCommandLine

import numpy
if numpy.version.version == '1.6.0':
    print 'WARNING: numpy 1.6.0 has a regression (http://projects.scipy.org/numpy/ticket/1870). Some commands wil not work. Please update to a newer version.'
    print ''

nv = numpy.version.version.split('.')
if len(nv) ==3:
    try:
        if int(nv[0]) == 1 and (int(nv[1]) <= 3 or (int(nv[1]) == 4 and int(nv[2]) == 0)):
            print 'WARNING: your numpy version (%s) is too old. Some commands will not work. Please update to a newer version.' % numpy.version.version
            print ''
    except ValueError:
        pass

rc_path = "~/.ibidas_rc"
rc_path = os.path.expanduser(rc_path)
global ipshell
global _rep_res
_rep_res = None

def complete_show(line, matches, longest_match):
    global _rep_res
    ipshell.IP.write('\n')
    if _rep_res is None:
        ipshell.IP.write(str(matches))
        ipshell.IP.write('\n')
    else:
        contextobj, attr, words = _rep_res
        _rep_res = None
        slice_names = contextobj.Names
        slice_names_set = set(slice_names)
        
        rem_words = []
        pos_slices = []
        for word in words:
            if word in slice_names_set:
                pos_slices.append(slice_names.index(word))
            else:
                rem_words.append(word)
        if(pos_slices):
            pos_slices.sort()
            ipshell.IP.write(contextobj.Get(*pos_slices).Info.info)
            ipshell.IP.write('\n')
        if(rem_words):
            ipshell.IP.write(str(rem_words))
            ipshell.IP.write('\n')
        _rep_res = None
        
   
    #well, this part was well documented, not....
    realline = readline.get_line_buffer()
    ipshell.IP.interact_prompt()
    ipshell.IP.write(realline[:readline.get_endidx()])
    addline = realline[readline.get_begidx() + len(line):]
    addline += chr(8) * len(addline)
    ipshell.IP.write(addline)



def rep_completer(self,event, line = None):
    global _rep_res
    if line is None:
        line = readline.get_line_buffer()
    try:
        contextobj, attr = parseCommandLine(line, ipshell)
        if(not isinstance(contextobj, representor.Representor)):
            raise TryNext
        words = dir2(contextobj)
        if attr:
            words = [word for word in words if word.startswith(attr)]
        else:
            words = [word for word in words if not word.startswith("_")]
        _rep_res = (contextobj, attr, words)
        if attr:
            words = [event.symbol[:-len(attr)] + word for word in words]
        else:
            words = [event.symbol + word for word in words]
        return words
    except: 
        raise TryNext

if len(sys.argv) == 2:
    with open(sys.argv[1], 'r') as argv:
        sys.argv = [ x[:-1] for x in argv.readlines() ];
        execfile(sys.argv[0]);
    #ewith
else:
    if oldip:
        ipshell = IPShellEmbed([],banner='Welcome to the IBIDAS system',exit_msg='IBIDAS shutting down ...',rc_override={'cache_size':0, 'readline_omit__names':2})
        del ipshell.IP.user_ns['_']
        ipshell()
    else:
        #Hack to get 'In' to show in Ipython, instead of the builtin 'In' history.
        #Requires a) setting builtin In, b) user_ns to globals(), c) del ipshell.user_ns['In']
        #Probably a bit fragile..
        import __builtin__
        __builtin__.__dict__['In'] = In
        cfg = Config()
        cfg.InteractiveShellEmbed.cache_size = 0
        #rc_override={'cache_size':0, 'readline_omit__names':2}
        ipshell = InteractiveShellEmbed(config=cfg, user_ns=globals(), banner2='Welcome to the IBIDAS system', exit_msg='IBIDAS shutting down ...')
        del ipshell.user_ns['_']
        del ipshell.user_ns['In']
        ipshell()

#ipshell.IP.set_hook('complete_command', rep_completer, re_key = '.*')

#if(ipshell.IP.has_readline):
#    readline.set_completion_display_matches_hook(complete_show)
