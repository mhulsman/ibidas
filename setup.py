import sys

req_version = (2,6)
cur_version = sys.version_info

if not ((cur_version[0] > req_version[0]) or (cur_version[0] == req_version[0] and cur_version[1] >= req_version[1])):
    sys.stderr.write("Your python interpreter is too old. Ibidas needs at least Python 2.6. Please consider upgrading.\n")
    sys.exit(-1)

if cur_version[0] > 2:
    sys.stderr.write("Ibidas only works with Python 2.x, not 3.x. Use 'python2', 'python2.7', 'python-2.7', 'python27' or something similar instead of 'python' to run the setup script.\n");
    sys.stderr.write("In case easy_install is used, use easy_install2, easy_install-2.7, or something similar (you can usetab-completion to find the available options).\n");
    sys.stderr.write("If these are not available, it might be necessary to install a version of Python 2.x in your distribution. \n")
    sys.exit(-1)
    
import ez_setup
ez_setup.use_setuptools()

from setuptools import setup,find_packages,Extension
import distutils.sysconfig
import os
import os.path

#include_dir = distutils.sysconfig.get_python_lib() + "/numpy/core/include/"
#if(not os.path.isfile(os.path.join(include_dir, "numpy/arrayobject.h"))):
#    #print os.path.join(include_dir, "numpy/arrayobject.h")
#    include_dir = include_dir.replace('lib', 'lib64')
#    if(not os.path.isfile(os.path.join(include_dir, "numpy/arrayobject.h"))):
#        #print os.path.join(include_dir, "numpy/arrayobject.h")
#        raise RuntimeError('numpy array headers not found')

if not os.path.isdir('docs/_build'):
    os.mkdir('docs/_build')

setup(
    name="Ibidas",
    version="0.1.23",
    packages = find_packages(),
    test_suite = "test",
    scripts = ['bin/ibidas','bin/ibidas_shell.py','bin/python_2_please'],
     install_requires=['numpy>=1.4.1','numpy!=1.6.0','sqlalchemy>=0.6.4','ipython>=0.10.1','sphinx>=1.0.5'],
     author = "M. Hulsman & J.J.Bot",
     author_email = "m.hulsman@tudelft.nl",
     description = "Ibidas is an environment for data handling and exploration, able to cope with different data structures and sources",
     url = "https://trac.nbic.nl/ibidas/",
     license = "LGPLv2.1",

)

