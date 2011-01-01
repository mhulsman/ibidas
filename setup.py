#!/usr/bin/env python
import ez_setup
ez_setup.use_setuptools()

from setuptools import setup,find_packages,Extension
import distutils.sysconfig

include_dir = distutils.sysconfig.get_python_lib() + "/numpy/core/include"

setup(
    name="Ibidas",
    version="0.1.0",
    packages = find_packages(),
    test_suite = "test",
    ext_modules = [
        Extension(
    		'closure', 
    		['src/closure.c'],
    		extra_compile_args=["-Wall"]
	    ),
        Extension(
		    'multi_visitor', 
		    ['src/multi_visitor.c'],
		    extra_compile_args=["-Wall"]
	    ),
        Extension(
            'base_container', 
            ['src/base_container.c'],
            include_dirs=[include_dir],
            extra_compile_args=["-Wall"]
        ),
        Extension(
            'cutils', 
            ['src/SFMT.c','src/cutils.c'], 
            include_dirs=[include_dir],
            extra_compile_args=["-Wall", "-ggdb3", "-O3", "-msse2", "-DHAVE_SSE2","-DMEXP=19937"]
#            extra_compile_args=["-Wall", "-O3", "-msse2","-msse4.2", "-DHAVE_SSE2","-DMEXP=19937"]
        ),
     ],
     install_requires=['numpy','sqlalchemy','ipython'],


)

