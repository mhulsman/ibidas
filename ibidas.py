#!/usr/bin/env python
import sys
import getopt
import getpass
from IPython.Shell import IPShellEmbed
import config

_ConfigDefault = {
    "system.tmp_dir":"/tmp"
}

if(__name__ == '__main__'):
    config = config.loadConfig('ibidas.ini', _ConfigDefault)
    #sys.argv = sys.argv[0] # remove commandline arguments so ipython doesn't see them
    
    ipshell = IPShellEmbed(banner='Welcome to the IBIDAS system',exit_msg='IBIDAS shutting down ...')
    del ipshell.IP.user_ns['_']
    ipshell()
