#include "Python.h"
#include "structmember.h"

#include "closure.h"

#define MVISITOR_MODULE
#include "multi_visitor.h"


/* Function returning object which can be mapped 
 * to visit function by findmethod (and used as key 
 * to cache this result) */
static PyObject * _visitkey(PyObject * self, PyObject ** visited)/*{{{*/
{
    PyObject * res;
    res = (PyObject *) (*visited)->ob_type;
    Py_INCREF(res);
    return (PyObject *) res;
}/*}}}*/

static PyObject * _direct_visitkey(PyObject * self, PyObject ** visited)/*{{{*/
{
    Py_INCREF(*visited);
    return (PyObject *) *visited;
}/*}}}*/

/* Function to find method in self, using prefix and visitkey 
 * obj. This specific function assumes obj is an TypeObject and
 * walks through its MRO to find prefix<mro-classname> functions 
 * in self*/
static PyObject * _findmethod(PyObject *self, PyObject *prefix, PyObject *visitkey)/*{{{*/
{
    PyObject * mro = PyObject_CallMethod(visitkey,"mro",NULL);    //visited->ob_type->tp_mro;  #fails for floats
    PyObject * cur_class;
    PyObject * curname;
    PyObject * visit_method;
    int i = 0;
    const char *s;

    if(mro == NULL || !PyList_Check(mro))
    {
        PyErr_SetString(PyExc_TypeError, "MRO error.");
        Py_XDECREF(mro);
        return 0;
    }
    

    Py_ssize_t mrolength = PyList_GET_SIZE(mro);
    for(i = 0; i < mrolength; i++)
    {
        cur_class = PyList_GET_ITEM(mro,i);

        /* Get class name */
        s = strrchr(((PyTypeObject *)cur_class)->tp_name, '.');
        if (s == NULL)
            s = ((PyTypeObject * )cur_class)->tp_name;
        else
            s++;

        /* Create method name */
        curname = PyString_FromFormat("%s%s",PyString_AS_STRING(prefix),s);
        if(curname == NULL)
        {
            Py_DECREF(mro);
            return 0;
        }
        visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
        Py_DECREF(curname);
        if(visit_method != NULL)
            break;
        else
            PyErr_Clear();
    }
    Py_DECREF(mro);
    
    return visit_method; 
}/*}}}*/


static PyObject * _direct_findmethod(PyObject *self, PyObject *prefix, PyObject *visitkey)/*{{{*/
{
    PyObject * curkey;
    PyObject * curname;
    PyObject * visit_method;
    const char *s;

    curkey = PyObject_Str(visitkey);
    s = PyString_AS_STRING(curkey);
    /* Create method name */
    curname = PyString_FromFormat("%s%s",PyString_AS_STRING(prefix),s);
    Py_DECREF(curkey);
    if(curname == NULL)
        return 0;

    visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
    Py_DECREF(curname);
    if(visit_method == NULL)
        PyErr_Clear();
    
    return visit_method; 
}/*}}}*/
/* Function to handle the case when no method is found. */
static PyObject * _notfound(PyObject *self, PyObject * prefix, PyObject *visited, long flags)/*{{{*/
{
    PyObject *curname, *visit_method = NULL, *tmp;
    switch(flags & (VISIT_NF_ROBJ | VISIT_NF_ELSE | VISIT_NF_ERROR))
    {
        case VISIT_NF_ROBJ:
                Py_INCREF(Py_None);
                visit_method = Py_None;
                break;
        case VISIT_NF_ELSE:
                curname = PyString_FromFormat("%s%s",PyString_AS_STRING(prefix),"else");
                if(curname == NULL)
                    return NULL;
                visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
                Py_DECREF(curname);
                if(visit_method != NULL)
                    break;
        default:
                tmp = PyObject_Repr((PyObject *)visited);
                if(tmp == NULL)
                    PyErr_SetString(PyExc_TypeError, "No suitable visit method found.");
                else
                {
                    PyErr_Format(PyExc_TypeError, "No suitable visit method found for %s,%s.",PyString_AsString(prefix),PyString_AsString(tmp));
                    Py_DECREF(tmp);
                }
    }
    return visit_method;
}/*}}}*/

/* Visit helper*/
static PyObject * _visit_withkey(PyObject *self, PyObject * visited, PyObject * visitkey, PyObject *args, PyObject *kwargs, PyObject *closure)/*{{{*/
{
    Py_ssize_t arglength;
    PyObject * visit_method, * visit_args;
    PyObject * res, * key;
    PyObject * prefix, *cache;
    PyObject ** dictptr, *cachekey = NULL; //cache related
    long flags,i;
    FindMethodFunction func_findmethod;
    NotFoundFunction func_notfound;

    /* Closure variables should have been checked during construction closure */
    prefix = PyTuple_GET_ITEM(closure,0);
    flags = PyInt_AS_LONG(PyTuple_GET_ITEM(closure,1));


    cache = self->ob_type->tp_dict;

    
    if(flags & VISIT_CACHE)
    {
        dictptr = _PyObject_GetDictPtr(self);
        if (dictptr != NULL) 
        {
            PyObject *dict = *dictptr;
            if(dict == NULL)
            {
                dict = PyDict_New();
                if(dict == NULL)
                    return NULL;
                *dictptr = dict;
            }

            cachekey = PyTuple_Pack(2,prefix,visited);
            res = PyDict_GetItem(dict, cachekey);
            if(res != NULL)
            {
                Py_INCREF(res);
                return res;
            }
	    }
    }
    key = PyTuple_Pack(2,prefix,visitkey);
    
    if(key == NULL)
        return NULL;
    visit_method = PyDict_GetItem(cache,key);


    if(visit_method == NULL)
    {
        func_findmethod = (FindMethodFunction) PyCObject_AsVoidPtr(PyTuple_GET_ITEM(closure,3));
        visit_method = func_findmethod(self,prefix,visitkey);
         

        if(visit_method == NULL)
        {
            /* NULL can mean not found or error */
            if(PyErr_Occurred())
            {
                Py_DECREF(key);
                return NULL;
            }
            func_notfound = (NotFoundFunction) PyCObject_AsVoidPtr(PyTuple_GET_ITEM(closure,4));
            visit_method = func_notfound(self,prefix,visited,flags);
            if(visit_method == NULL)
            {
                Py_DECREF(key);
                return NULL;
            }

        }
        if(PyDict_SetItem(cache,key,visit_method) < 0)
        {
            Py_DECREF(visit_method);
            Py_DECREF(key);
            return NULL;
        }
    }
    else
        Py_INCREF(visit_method);
    Py_DECREF(key);

    if(visit_method == Py_None)
    {
        Py_DECREF(visit_method);
        Py_INCREF(visited);
        return visited;
    }

    if(!PyCallable_Check(visit_method))
    {
        PyErr_SetString(PyExc_TypeError, "Found visit method is not callable.");
        Py_DECREF(visit_method);
        return 0;
    }
     
    arglength = PyTuple_GET_SIZE(args);
    visit_args = PyTuple_New(arglength + 2);
    if(visit_args == NULL)
    {
        Py_DECREF(visit_method);
        return 0;
    }
    
    Py_INCREF(self);
    PyTuple_SET_ITEM(visit_args,0,(PyObject *)self);
    Py_INCREF(visited);
    PyTuple_SET_ITEM(visit_args,1,(PyObject *)visited); //could have been replaced
     
    for(i = 0; i < arglength; i++)
    {
        PyObject * tmp = PyTuple_GET_ITEM(args,i);
        Py_XINCREF(tmp);
        PyTuple_SET_ITEM(visit_args,i + 2,tmp);
    }

    res = PyObject_Call(visit_method,visit_args,kwargs);
    Py_DECREF(visit_method);
    Py_DECREF(visit_args);


    if((flags & VISIT_CACHE) && res) //should we cache the result?
    {
        if(!(cachekey && dictptr)) //can we cache the result?
        {
            PyErr_SetString(PyExc_TypeError, "Caching failed.");
            return NULL;
        }
        PyObject *dict = *dictptr;
        if (dict == NULL) //if the object has no dict yet, create one
        {
			dict = PyDict_New();
            if(dict == NULL)
            {
                Py_DECREF(res);
                return NULL;
            }
			*dictptr = dict;
		}
        if(PyDict_SetItem(dict, cachekey,res) < 0)
        {
            Py_DECREF(res);
            return NULL;
        }
    }

    return res;
}/*}}}*/

/* Visit function called when an visit operation is performed.
 * Search in cache, if not found, uses findmethod to find method
 * to handle the visit request. Closure object should contain
 * (prefix[PyString],flags[PyInt],visitkeyfunc[PyCobject],findmethodfunc[PyCObject])*/
static PyObject * visit(PyObject *self, PyObject *args, PyObject *kwargs, PyObject *closure)/*{{{*/
{
    Py_ssize_t arglength;
    PyObject * visited, *visitkey;
    PyObject * res, *nargs;
    PyObject * prefix;
    VisitKeyFunction func_visitkey;

    /* Closure variables should have been checked during construction closure */
    prefix = PyTuple_GET_ITEM(closure,0);
    func_visitkey = (VisitKeyFunction) PyCObject_AsVoidPtr(PyTuple_GET_ITEM(closure,2));

    arglength = PyTuple_GET_SIZE(args);
    if(arglength == 0)
    {
        PyErr_SetString(PyExc_TypeError, "Visit method should be called with the object to visit.");
        return 0;
    }
    
    visited = PyTuple_GET_ITEM(args,0);
    if(visited == NULL)
        return 0;
    
    visitkey = func_visitkey(self,&visited);
    if(visitkey == NULL)
        return NULL;

    nargs = PyTuple_GetSlice(args, 1, arglength);

    res = _visit_withkey(self, visited, visitkey, nargs, kwargs, closure);

    Py_DECREF(nargs);
    Py_DECREF(visitkey);

    return res;
}/*}}}*/


/* Visit key function. Similar to visit, but user can explicity
 * give key to use
*/
static PyObject * visitKey(PyObject *self, PyObject *args, PyObject *kwargs, PyObject *closure)/*{{{*/
{
    Py_ssize_t arglength;
    PyObject * visited, *visitkey;
    PyObject * res, *nargs;
    PyObject * prefix;

    /* Closure variables should have been checked during construction closure */
    prefix = PyTuple_GET_ITEM(closure,0);

    arglength = PyTuple_GET_SIZE(args);
    if(arglength <= 1)
    {
        PyErr_SetString(PyExc_TypeError, "Visit method should be called with the object to visit.");
        return 0;
    }
    
    visited = PyTuple_GET_ITEM(args,1);
    if(visited == NULL)
        return 0;
    
    visitkey = PyTuple_GET_ITEM(args,0);
    if(visitkey == NULL)
        return NULL;

    nargs = PyTuple_GetSlice(args, 2, arglength);

    res = _visit_withkey(self, visited, visitkey, nargs, kwargs, closure);

    Py_DECREF(nargs);

    return res;
}/*}}}*/



static PyMethodDef object_methods[] = {
    {"visit",(PyCFunction) visit, METH_VARARGS | METH_KEYWORDS, "Visitor method"},
    {"visitKey",(PyCFunction) visitKey, METH_VARARGS | METH_KEYWORDS, "Visitor key method"},
    {NULL}  /* Sentinel */
};

/* Function to modify a type to include the visit function */
static int _visitify(PyObject * prefix, /*{{{*/
    PyTypeObject * type, PyObject * flags, PyObject *closure)
{

    PyObject *dict, *descr, *nclosure, *vkfuncobj, *fmfuncobj, *nffuncobj;
    PyObject * prefixKey = prefix;
    PyObject *descrKey;

    if(!PyString_Check(prefix))
    {
        PyErr_SetString(PyExc_TypeError, "Prefixes should be strings.");
        return -1;
    }
    dict = type->tp_dict;
    if(dict == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Type not correctly initialized (no dict)");
        return -1;
    }

    vkfuncobj = PyTuple_GET_ITEM(closure,1);
    fmfuncobj = PyTuple_GET_ITEM(closure,2);
    nffuncobj = PyTuple_GET_ITEM(closure,3);
    nclosure = PyTuple_Pack(5,prefix,flags,vkfuncobj,fmfuncobj,nffuncobj);
    if(nclosure == NULL)
        return -1;

    descr = PyDescr_NewClosureMethod(type,&object_methods[0],nclosure);
    descrKey = PyDescr_NewClosureMethod(type,&object_methods[1],nclosure);
    Py_DECREF(nclosure);
    if(descr == NULL || descrKey == NULL)
        return -1;

    
    Py_INCREF(prefixKey);
    PyString_ConcatAndDel(&prefixKey, PyString_FromString("Key"));
    if(prefixKey == NULL)
    {
        Py_DECREF(descr);
        Py_DECREF(descrKey);
        return -1;
    }

    if(PyDict_SetItem(dict,prefix,descr)<0)
    {
        Py_DECREF(descr);
        Py_DECREF(descrKey);
        return -1;
    }
    if(PyDict_SetItem(dict,prefixKey,descrKey)<0)
    {
        Py_DECREF(descr);
        Py_DECREF(descrKey);
        return -1;
    }
    return 0;
}/*}}}*/


/* Function creating a class and adding visit functions as requested.
 * Parameters: name -> class name
 *             prefixes -> visit function prefixes (i.e. 'visit')
 *             bases -> base clases (tuple)
 *             flags -> flags for visit function
 */
static PyObject * VisitorFactory(PyObject * self, PyObject *args, PyObject *kwds, PyObject *closure) /*{{{*/
{
    PyObject *name = NULL, *prefixes = NULL, *bases = NULL, *flags = NULL, *typedict = NULL, *tmp;
    PyObject *nargs, *res = NULL;
    PyTypeObject *ntype;
    long i,prefix_size;
    VisitifyFunction vsfunc = (VisitifyFunction) PyCObject_AsVoidPtr(PyTuple_GET_ITEM(closure,0));

    static char *kwlist[] = {"name","prefixes","bases","flags","typedict",NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O!O!O!O!O!", kwlist, &PyString_Type,&name,(PyObject *)&PyTuple_Type, &prefixes, &PyTuple_Type, &bases, (PyObject *)&PyInt_Type, &flags, (PyObject *)&PyDict_Type, &typedict)) 
        return NULL;

    /* Setting default values if parameters not set */
    if(name == NULL)
    {
        name = PyString_FromString("Visitor");
        if(name == NULL)
            return NULL;
    }
    else
        Py_INCREF(name);

    if(prefixes == NULL)
    {
        tmp = PyString_FromString("visit");
        if(tmp == NULL)
            return NULL;
        prefixes = Py_BuildValue("(N)",tmp);
        if(prefixes == NULL)
        {
            Py_DECREF(tmp);
            return NULL;
        }
    }
    else
        Py_INCREF(prefixes);

    if(bases == NULL)
    {
      bases = PyTuple_Pack(1,&PyBaseObject_Type);
      if(bases == NULL)
      {
        Py_DECREF(prefixes);
        return NULL;
      }
    }
    else
       Py_INCREF(bases);

    
    if(flags == NULL)
    {
        flags = PyInt_FromLong(0);
        if(flags == NULL)
        {
            Py_DECREF(prefixes);
            Py_DECREF(bases);
            return NULL;
        }
    }
    else
        Py_INCREF(flags);
  
    if(typedict == NULL)
        typedict = PyDict_New();
    else
        Py_INCREF(typedict);

    nargs = Py_BuildValue("(OON)",name,bases,typedict);
    if(nargs == NULL)
        goto vis_error;
    ntype = (PyTypeObject *) PyType_Type.tp_new(&PyType_Type,nargs,NULL);
    Py_DECREF(nargs);
    if(ntype == NULL)
        goto vis_error;


    prefix_size = PyTuple_GET_SIZE(prefixes);
    for(i = 0; i < prefix_size; i++)
    {
        tmp = PyTuple_GET_ITEM(prefixes,i);
        if(vsfunc(tmp,ntype,flags,closure) < 0)
        {
            Py_DECREF(ntype);
            goto vis_error;
        }
    }
    
    res = (PyObject *) ntype;
vis_error:
    Py_DECREF(name);
    Py_DECREF(flags);
    Py_DECREF(bases);
    Py_DECREF(prefixes);
    return res;
}/*}}}*/


static PyMethodDef closure_methods[] = {
    {"VisitorFactory",(PyCFunction) VisitorFactory, METH_KEYWORDS, "Creates visitor classes"},
    {NULL}  /* Sentinel */
};

static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};


#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initmulti_visitor(void) 
{
    PyObject* m;
    PyObject *c_api_object;
    PyObject *closure, *direct_closure;
    PyObject *visitfactoryfunc, *direct_visitfactoryfunc;
    static void *PyMVisitor_API[PyMVisitor_API_pointers];
   
    m = Py_InitModule3("multi_visitor", module_methods,
                       "Visitor module");
    if (m == NULL)
      return;
    if(import_closure()<0)
    {
        printf("import_closure failed!\n");
        return;
    }

    //Flags for VisitFactory function
    PyModule_AddObject(m, "NF_ERROR", PyInt_FromLong(VISIT_NF_ERROR));
    PyModule_AddObject(m, "NF_ROBJ", PyInt_FromLong(VISIT_NF_ROBJ));
    PyModule_AddObject(m, "NF_ELSE", PyInt_FromLong(VISIT_NF_ELSE));
    PyModule_AddObject(m, "F_CACHE", PyInt_FromLong(VISIT_CACHE));

    /* Add closure funciton object for visitfactory function */
    //replaceable functions used by the visit machinery
    closure = create_closure(_visitkey, _findmethod,
                    _visitify, _notfound);
    if(closure == NULL)
        return;
    visitfactoryfunc = PyClosureFunction_New(&closure_methods[0],m,m,closure);
    Py_DECREF(closure);
    if(visitfactoryfunc == NULL)
       return;
    
    //add to module
    PyModule_AddObject(m,"VisitorFactory",visitfactoryfunc);
     
     /* Add closure funciton object for visitfactory function */
    //replaceable functions used by the visit machinery
    direct_closure = create_closure(_direct_visitkey, _direct_findmethod,
                    _visitify, _notfound);
    if(direct_closure == NULL)
        return;
    direct_visitfactoryfunc = PyClosureFunction_New(&closure_methods[0],m,m,direct_closure);
    Py_DECREF(direct_closure);
    if(direct_visitfactoryfunc == NULL)
       return;
    PyModule_AddObject(m,"DirectVisitorFactory",direct_visitfactoryfunc);
  
    PyMVisitor_API[Visitor_factory_NUM] = (void *)VisitorFactory;
    PyMVisitor_API[Visitor_visitkey_NUM] = (void *)_visitkey;
    PyMVisitor_API[Visitor_findmethod_NUM] = (void *)_findmethod;
    PyMVisitor_API[Visitor_visitify_NUM] = (void *)_visitify;
    PyMVisitor_API[Visitor_notfound_NUM] = (void *)_notfound;
    PyMVisitor_API[Visitor_visit_NUM] = (void *)visit;
    
    /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)PyMVisitor_API, NULL);
    
    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);
    
}
