#include <Python.h>
#include "structmember.h"
#include <string.h>



static PyTypeObject VisitorType;

typedef struct {
    PyObject_HEAD
    PyObject * vcache;
} Visitor;

static void
Visitor_dealloc(Visitor* self)
{
    Py_XDECREF(self->vcache);
    self->ob_type->tp_free((PyObject*)self);
}

static int
Visitor_init(Visitor *self, PyObject *args, PyObject *kwds)
{
    PyObject *key, *vcache;
    if(self->vcache == NULL)
    {
        key = PyString_FromString("__cache__");
        if(key == NULL)
            return -1;
        vcache = PyDict_GetItem(self->ob_type->tp_dict,key);
        if(vcache == NULL || !PyDict_Check(vcache))
        {
            vcache = PyDict_New();
            if(vcache == NULL)
                goto error;
            if(PyDict_SetItem(self->ob_type->tp_dict,key,vcache) == -1)
                goto error;
        }
        else
        {
            Py_INCREF(vcache);
        }
        self->vcache = vcache;
        Py_DECREF(key);
    }
    return 0;
 error:
    Py_XDECREF(vcache);
    Py_DECREF(key);
    return -1;
    
}

static PyObject * Visitor_visit(Visitor *self,PyObject *args)
{
    Py_ssize_t arglength;
    PyObject * visited;
    PyObject * visit_method;
    PyObject * visit_args;
    PyObject * res;
    int i = 0;

    if(self->vcache == NULL)
         if(Visitor_init(self,NULL,NULL) == -1)
            return 0;
    
    if(!PyTuple_Check(args))
    {
        PyErr_SetString(PyExc_TypeError, "Visit method should be called with the object to visit.");
        return 0;
    }

    arglength = PyTuple_GET_SIZE(args);
    if(arglength == 0)
    {
        PyErr_SetString(PyExc_TypeError, "Visit method should be called with the object to visit.");
        return 0;
    }
    
    visited = PyTuple_GET_ITEM(args,0);
    if(visited == NULL)
        return 0;

    visit_method = PyDict_GetItem(self->vcache,(PyObject *) visited->ob_type);
    if(visit_method == NULL)
    {
        PyObject * mro = PyObject_CallMethod((PyObject *)visited->ob_type,"mro",NULL);    //visited->ob_type->tp_mro;  #fails for floats
        PyObject * cur_class;
        PyObject * curname;
        const char *s;
        char *sn;
        int name_length;

        if(mro == NULL || !PyList_Check(mro))
        {
            PyErr_SetString(PyExc_TypeError, "MRO error.");
            Py_XDECREF(mro);
            return 0;
        }
        
        sn = PyMem_Malloc(sizeof(char) * 1024);
        if(sn == NULL)
            return 0;


        Py_MEMCPY(sn,"visit",6);

        Py_ssize_t mrolength = PyList_GET_SIZE(mro);
        for(i = 0; i < mrolength; i++)
        {
            cur_class = PyList_GET_ITEM(mro,i);
            s = strrchr(((PyTypeObject *)cur_class)->tp_name, '.');
            if (s == NULL)
                s = ((PyTypeObject * )cur_class)->tp_name;
            else
                s++;
            name_length = strlen(s);

            if(name_length > 1018)
            {
                PyErr_SetString(PyExc_TypeError, "Visit class name too long.");
                PyMem_Free(sn);
                return 0;
            }
            Py_MEMCPY(&sn[5],s,name_length + 1);
            curname = PyString_FromString(sn);
            if(curname == NULL)
            {
                PyMem_Free(sn);
                return 0;
            }

            if(PyObject_HasAttr((PyObject *) self->ob_type,curname))
            {
                visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
                Py_DECREF(curname);
                if(visit_method == NULL ||
                    PyDict_SetItem(self->vcache,(PyObject *) visited->ob_type,visit_method) == -1)
                {
                    Py_XDECREF(visit_method);
                    PyMem_Free(sn);
                    return 0;
                }
                break;
            }
            Py_DECREF(curname);
        }
        PyMem_Free(sn);
        Py_DECREF(mro);
        if(visit_method == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "No suitable visit method found.");
            return 0;
        }
    }
    else
        Py_INCREF(visit_method);
   
    
    if(!PyCallable_Check(visit_method))
    {
        PyErr_SetString(PyExc_TypeError, "Cached visit method is not callable.");
        Py_DECREF(visit_method);
        return 0;
    }
     
    visit_args = PyTuple_New(arglength+1);
    if(visit_args == NULL)
    {
        Py_DECREF(visit_method);
        return 0;
    }
    
    Py_INCREF(self);
    PyTuple_SET_ITEM(visit_args,0,(PyObject *)self);
     
    for(i = 0; i < arglength; i++)
    {
        PyObject * tmp = PyTuple_GET_ITEM(args,i);
        Py_XINCREF(tmp);
        PyTuple_SET_ITEM(visit_args,i+1,tmp);
    }

    res = PyObject_CallObject(visit_method,visit_args);
    Py_DECREF(visit_method);
    Py_DECREF(visit_args);
    return res;
}

static PyMethodDef Visitor_methods[] = {
    {"visit", (PyCFunction)Visitor_visit, METH_VARARGS,
     "Visit class by calling appropiate method"
    },
    {NULL} 
};


static PyTypeObject VisitorType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "visitor.Visitor",             /*tp_name*/
    sizeof(Visitor),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Visitor_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                          /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_CLASS, /*tp_flags*/
    "Visitor object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Visitor_methods,         /* tp_methods */
    0,                         /* tp_members */
    0,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Visitor_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};

static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initvisitor(void) 
{
    PyObject* m;

    VisitorType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&VisitorType) < 0)
        return;

    m = Py_InitModule3("visitor", module_methods,
                       "Example module that creates an extension type.");

    if (m == NULL)
      return;

    Py_INCREF(&VisitorType);
    PyModule_AddObject(m, "Visitor", (PyObject *)&VisitorType);
}
