"""Module implementing loading/writing of configuration 
variables to ini file

Author: Marc Hulsman
Date: 20-feb-08"""

import ConfigParser
import string
import os
import os.path

def loadConfig(rc_path, config={}):
    """
    returns a dictionary with key's of the form
    <section>.<option> and the values 
    """
    config = config.copy()
    if(os.path.isfile(rc_path)):    
        cp = ConfigParser.ConfigParser()
        cp.read(rc_path)
        for sec in cp.sections():
            name = string.lower(sec)
            for opt in cp.options(sec):
                config[name + "." + string.lower(opt)] = string.strip(cp.get(sec, opt))
    return config

def writeConfig(filename, config):
    """
    given a dictionary with key's of the form 'section.option: value'
    write() generates a list of unique section names
    creates sections based that list
    use config.set to add entries to each section
    """
    cp = ConfigParser.ConfigParser()
    sections = set([k.split('.')[0] for k in config.keys()])
    map(cp.add_section, sections)
    for k,v in config.items():
        s, o = k.split('.')
        cp.set(s, o, v)
    cp.write(open(filename, "w"))

''' 
When more than one value is needed from a config file in a script, please use the loadConfig method.
This method is just a shortcut when only one specific value is of interest.
'''
def getValue(file, config, value):
    cfg = loadConfig(file, config)
    return cfg[value]


_ConfigDefault = {
    "system.tmp_dir":"/tmp"
}

config = loadConfig(os.path.expanduser('~/.ibidas/ibidas.cfg'), _ConfigDefault)
