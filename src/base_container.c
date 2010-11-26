#include <Python.h>
#include "structmember.h"
#include "helper.h"
#include "numpy/arrayobject.h"

#define BASECONTAINER_MODULE
#include "base_container.h"

/*{{{*/
static void
ModificationSemaphore_dealloc(ModificationSemaphore* self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static int
ModificationSemaphore_init(ModificationSemaphore *self, PyObject *args, PyObject *kwds)
{
   PyObject *nrmod;
   

   static char *kwlist[] = {"nrmod",NULL};
   if(!PyArg_ParseTupleAndKeywords(args,kwds,"O!",kwlist,
            &PyInt_Type,&nrmod))
      return -1;
   self->nrmod = PyInt_AsLong(nrmod);
   return 0;
}

static PyObject *
ModificationSemaphore_signal(ModificationSemaphore *self)
{
   self->curid++;
   Py_INCREF(Py_None);
   return Py_None;
}

static PyMethodDef ModificationSemaphore_methods[] = {
    {"signal", (PyCFunction)ModificationSemaphore_signal, METH_NOARGS,
     "Signal that something has changed"
    },
    {NULL} 
};

static PyObject *
ModificationSemaphore_getNrMod(ModificationSemaphore *self, void *closure)
{
    return PyInt_FromLong(self->nrmod);
}
static int
ModificationSemaphore_setNrMod(ModificationSemaphore *self, PyObject *value, void *closure)
{
    if(!PyInt_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Semaphore state should be an integer object");
        return -1;
    }
    self->nrmod = PyInt_AsLong(value);
    return 0;
}

static PyObject *
ModificationSemaphore_getCurID(ModificationSemaphore *self, void *closure)
{
    return PyInt_FromLong(self->curid);
}
static int
ModificationSemaphore_setCurID(ModificationSemaphore *self, PyObject *value, void *closure)
{
    if(!PyInt_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Semaphore state should be an integer object");
        return -1;
    }
    self->curid = PyInt_AsLong(value);
    return 0;
}

static PyGetSetDef ModificationSemaphore_getseters[] = {
    {"nrmod", 
     (getter)ModificationSemaphore_getNrMod, (setter)ModificationSemaphore_setNrMod,
     "Nr of modifiables using this semaphore",
     NULL},
    {"curid", 
     (getter)ModificationSemaphore_getCurID, (setter)ModificationSemaphore_setCurID,
     "Current semaphore state",
     NULL},
    {NULL}  /* Sentinel */
};


static PyTypeObject ModificationSemaphore_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "base_container.ModificationSemaphore",             /*tp_name*/
    sizeof(ModificationSemaphore),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)ModificationSemaphore_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,          /*tp_call*/
    0,                         /*tp_str*/
    0,                          /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_CLASS, /*tp_flags*/
    "ModificationSemaphore object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    ModificationSemaphore_methods,         /* tp_methods */
    0,                         /* tp_members */
    ModificationSemaphore_getseters,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)ModificationSemaphore_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};/*}}}*/

/*{{{*/
static void
QueryResult_dealloc(QueryResult* self)
{
    Py_XDECREF(self->data);
    Py_XDECREF(self->shape);
    Py_XDECREF(self->fields);
    Py_XDECREF(self->modify_check);
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
QueryResult_repr(QueryResult* self)
{
    PyObject *shape;
    long nfields;
    char * cshape;
    if(self->shape != NULL && self->fields != NULL)
    {
        shape = PyObject_Repr(self->shape);
        nfields = PySequence_Length(self->fields);
        
        if(shape != NULL)
        {
            cshape = PyString_AsString(shape);
            return PyString_FromFormat("QR[%s,%ld]",cshape,nfields); 
        }
    }
    else
        PyErr_SetString(PyExc_TypeError, "QueryResult object has no valid shape or fields.");
    return NULL;
}

static int
QueryResult_init(QueryResult *self, PyObject *args, PyObject *kwds)
{
   PyObject *data,*shape,*fields,*tmp;
   

   static char *kwlist[] = {"data","shape","fields",NULL};
   if(!PyArg_ParseTupleAndKeywords(args,kwds,"OO!O!",kwlist,
            &data,&PyList_Type,&shape,&PyList_Type,&fields))
      return -1;

   if(!(PyTuple_Check(data)))
   {
       PyErr_SetString(PyExc_TypeError, "Data objects should be a tuple");
       return -1;
   }
   tmp = self->data;
   Py_INCREF(data);
   self->data = data;
   Py_XDECREF(tmp);
   
   tmp = self->shape;
   Py_INCREF(shape);
   self->shape = shape;
   Py_XDECREF(tmp);

   tmp = self->fields;
   Py_INCREF(fields);
   self->fields = fields;
   Py_XDECREF(tmp);

   self->cacheable = 0;
   return 0;
}

static PyObject *
QueryResult_call(PyObject *self, PyObject *args, PyObject *kwds)
{
    Py_INCREF(self);
    return self;
}

static PyObject *
QueryResult_copy(QueryResult *self)
{
   QueryResult *res = PyObject_New(QueryResult,&QueryResult_Type);
   if(res == NULL)
      return NULL;

   res->data = self->data;
   res->shape = PyList_Copy(self->shape);
   res->fields = PyList_Copy(self->fields);
   res->modify_check = self->modify_check;
   res->sem_state = self->sem_state;
   res->cacheable = self->cacheable;

   Py_XINCREF(res->data);
   Py_XINCREF(res->shape);
   Py_XINCREF(res->fields);
   Py_XINCREF(res->modify_check);
   
   return (PyObject *)res;
}

static PyObject *
QueryResult_isvalid(QueryResult *self)
{
   if(self->modify_check == NULL || ((ModificationSemaphore * )self->modify_check)->curid == self->sem_state)
   {
       Py_INCREF(Py_True);
       return Py_True;
   }
   else
   {
       Py_INCREF(Py_False);
       return Py_False;
   }
}

static PyMethodDef QueryResult_methods[] = {
    {"__copy__", (PyCFunction)QueryResult_copy, METH_NOARGS,
     "Returns a copy of this object"
    },
    {"isvalid", (PyCFunction)QueryResult_isvalid, METH_NOARGS,
     "Returns if the dataset is still valid (i.e. if its base data has not been changed)"
    },
    {NULL} 
};

static PyObject *
QueryResult_getModifyCheck(QueryResult *self, void *closure)
{
    if(self->modify_check == NULL)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }
    else
    {
        Py_INCREF(self->modify_check);
        return self->modify_check;
    }
}
static int
QueryResult_setModifyCheck(QueryResult *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == Py_None)
        return 0;

    if(value->ob_type != &ModificationSemaphore_Type)
    {
        PyErr_SetString(PyExc_TypeError, "modify_check should be an ModificationSemaphore");
        return -1;
    }

    tmp = self->modify_check;
    Py_INCREF(value);
    self->modify_check = value;
    Py_XDECREF(tmp);
    self->sem_state = ((ModificationSemaphore *) value)->curid;

    return 0;
}

static PyObject *
QueryResult_getSemState(QueryResult *self, void *closure)
{
    return PyInt_FromLong(self->sem_state);
}
static int
QueryResult_setSemState(QueryResult *self, PyObject *value, void *closure)
{
    if(!PyInt_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Semaphore state should be an integer object");
        return -1;
    }
    self->sem_state = PyInt_AsLong(value);
    return 0;
}
static PyObject *
QueryResult_getData(QueryResult *self, void *closure)
{
    if(self->data == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QueryResult object not initialized.");
        return NULL;
    }   
    Py_INCREF(self->data);
    return self->data;
}
static int
QueryResult_setData(QueryResult *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(!(PyTuple_Check(value)))
    {
        PyErr_SetString(PyExc_TypeError, "Data objects should be tuples");
        return -1;
    }
    tmp = self->data;
    Py_INCREF(value);
    self->data = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
QueryResult_getShape(QueryResult *self, void *closure)
{
    if(self->shape == NULL)
    {
        self->shape = PyList_New(0);
    }   
    Py_INCREF(self->shape);
    return self->shape;
}
static int
QueryResult_setShape(QueryResult *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(!(PyList_Check(value)))
    {
        PyErr_SetString(PyExc_TypeError, "Shape should be a list");
        return -1;
    }
    tmp = self->shape;
    Py_INCREF(value);
    self->shape = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
QueryResult_getFields(QueryResult *self, void *closure)
{
    if(self->fields == NULL)
    {
        self->fields = PyList_New(0);
    }   
    Py_INCREF(self->fields);
    return self->fields;
}
static int
QueryResult_setFields(QueryResult *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(!(PyList_Check(value)))
    {
        PyErr_SetString(PyExc_TypeError, "Fields should be a list");
        return -1;
    }
    tmp = self->fields;
    Py_INCREF(value);
    self->fields = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
QueryResult_getNField(QueryResult *self, void *closure)
{
    if(self->fields == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QueryResult object not initialized.");
        return NULL;
    }   
    return PyInt_FromLong(PySequence_Length(self->fields));
}

static PyObject *
QueryResult_getNDim(QueryResult *self, void *closure)
{
    if(self->shape == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QueryResult object not initialized.");
        return NULL;
    }   
    return PyInt_FromLong(PySequence_Length(self->shape));
}

static PyObject *
QueryResult_getCacheable(QueryResult *self, void *closure)
{
    if(self->cacheable == 1)
    {
        Py_INCREF(Py_True);
        return Py_True;
    }   
    else
    {
        Py_INCREF(Py_False);
        return Py_False;
    }
}
static int
QueryResult_setCacheable(QueryResult *self, PyObject *value, void *closure)
{
    if(!PyBool_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Cacheable parameter should be bool");
        return -1;
    }
    if(value == Py_True)
        self->cacheable = 1;
    else
        self->cacheable = 0;
    return 0;
}
static PyGetSetDef QueryResult_getseters[] = {
    {"data", 
     (getter)QueryResult_getData, (setter)QueryResult_setData,
     "Data object for this result object",
     NULL},
    {"shape", 
     (getter)QueryResult_getShape, (setter)QueryResult_setShape,
     "dimensional shape",
     NULL},
    {"fields", 
     (getter)QueryResult_getFields, (setter)QueryResult_setFields,
     "container fields",
     NULL},
    {"ndim", 
     (getter)QueryResult_getNDim, NULL,
     "Number of dimensions",
     NULL},
    {"nfield", 
     (getter)QueryResult_getNField, NULL,
     "Number of fields",
     NULL},
    {"modify_check", 
     (getter)QueryResult_getModifyCheck, (setter)QueryResult_setModifyCheck,
     "Semaphore",
     NULL},
    {"sem_state", 
     (getter)QueryResult_getSemState, (setter)QueryResult_setSemState,
     "Semaphore creation state",
     NULL},
    {"cacheable", 
     (getter)QueryResult_getCacheable, (setter)QueryResult_setCacheable,
     "Set to indicate if the source of this query result object could produce other or modified \
     query result objects",
     NULL},
    {NULL}  /* Sentinel */
};


static PyTypeObject QueryResult_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "base_container.QueryResult",             /*tp_name*/
    sizeof(QueryResult),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)QueryResult_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc) QueryResult_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    (ternaryfunc) QueryResult_call,          /*tp_call*/
    0,                         /*tp_str*/
    0,                          /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_CLASS, /*tp_flags*/
    "QueryResult object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    QueryResult_methods,         /* tp_methods */
    0,                         /* tp_members */
    QueryResult_getseters,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)QueryResult_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};/*}}}*/

/*{{{*/
static void Invar_dealloc(Invar* self)
{
    Py_XDECREF(self->descriptors);
    Py_XDECREF(self->actidx_descriptors);
    Py_XDECREF(self->fields);
    Py_XDECREF(self->actidx);
    Py_XDECREF(self->dims);
    Py_XDECREF(self->actidx_dims);
    Py_XDECREF(self->segments);
    Py_XDECREF(self->modify_sems);
    Py_XDECREF(self->select_query);
    Py_XDECREF(self->cache);
    Py_XDECREF(self->cap_class);
    Py_XDECREF(self->local);
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject * Invar_getDescriptors(Invar *self, void *closure)/*{{{*/
{
    if(self->descriptors == NULL)
    {
        Py_RETURN_NONE;
    }   
    Py_INCREF(self->descriptors);
    return self->descriptors;
}
static int Invar_setDescriptors(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        Py_XDECREF(self->descriptors);
        self->descriptors = NULL;
        return 0;
    }

    if(!(value == Py_None || (PyTuple_Check(value))))
    {
        PyErr_SetString(PyExc_TypeError, "Descriptors objects should be a tuple");
        return -1;
    }
    tmp = self->descriptors;
    Py_INCREF(value);
    self->descriptors = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getActIdxDescriptors(Invar *self, void *closure)/*{{{*/
{
    if(self->actidx_descriptors == NULL)
    {
        Py_RETURN_NONE;
    }   
    Py_INCREF(self->actidx_descriptors);
    return self->actidx_descriptors;
}
static int Invar_setActIdxDescriptors(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        Py_XDECREF(self->actidx_descriptors);
        self->actidx_descriptors = NULL;
        return 0;
    }
    if(!(value == Py_None || PyList_Check(value)))
    {
        PyErr_SetString(PyExc_TypeError, "Active descriptor index should be a list");
        return -1;
    }
    tmp = self->actidx_descriptors;
    Py_INCREF(value);
    self->actidx_descriptors = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getFields(Invar *self, void *closure)/*{{{*/
{
    if(self->fields == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar object not properly initialized (no fields).");
        return NULL;
    }   
    Py_INCREF(self->fields);
    return self->fields;
}
static int Invar_setFields(Invar *self, PyObject *value, void *closure)
{
    if(value == NULL)
    {
        Py_XDECREF(self->fields);
        self->fields = NULL;
        return 0;
    }
    
    PyObject *tmp;
    if(!PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Field objects should be a list");
        return -1;
    }
    tmp = self->fields;
    Py_INCREF(value);
    self->fields = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getActIdx(Invar *self, void *closure)/*{{{*/
{
    if(self->actidx == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar object not properly initialized (no actidx).");
        return NULL;
    }   
    Py_INCREF(self->actidx);
    return self->actidx;
}
static int Invar_setActIdx(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        Py_XDECREF(self->actidx);
        self->actidx = NULL;
        return 0;
    }
    if(!PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Active index should be a list");
        return -1;
    }
    tmp = self->actidx;
    Py_INCREF(value);
    self->actidx = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getDims(Invar *self, void *closure)/*{{{*/
{
    if(self->dims == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar object not properly initialized (no dims).");
        return NULL;
    }   
    Py_INCREF(self->dims);
    return self->dims;
}
static int Invar_setDims(Invar *self, PyObject *value, void *closure)
{
    if(value == NULL)
    {
        Py_XDECREF(self->dims);
        self->dims = NULL;
        return 0;
    }
    
    PyObject *tmp;
    if(!PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Dimensions should be a list");
        return -1;
    }
    tmp = self->dims;
    Py_INCREF(value);
    self->dims = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getActIdxDims(Invar *self, void *closure)/*{{{*/
{
    if(self->actidx_dims == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar object not properly initialized (no actidx_dims).");
        return NULL;
    }   
    Py_INCREF(self->actidx_dims);
    return self->actidx_dims;
}
static int Invar_setActIdxDims(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        Py_XDECREF(self->actidx_dims);
        self->actidx_dims = NULL;
        return 0;
    }
    if(!PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Active dim index should be a list");
        return -1;
    }
    tmp = self->actidx_dims;
    Py_INCREF(value);
    self->actidx_dims = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getSegments(Invar *self, void *closure)/*{{{*/
{
    if(self->segments == NULL)
    {
        Py_RETURN_NONE;
    }   
    Py_INCREF(self->segments);
    return self->segments;
}
static int Invar_setSegments(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        Py_XDECREF(self->segments);
        self->segments = NULL;
        return 0;
    }
    if(!(value == Py_None || PyTuple_Check(value)))
    {
        PyErr_SetString(PyExc_TypeError, "Segments should be None or a tuple");
        return -1;
    }
    tmp = self->segments;
    Py_INCREF(value);
    self->segments = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getModifySems(Invar *self, void *closure)/*{{{*/
{
    if(self->modify_sems == NULL)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }   
    Py_INCREF(self->modify_sems);
    return self->modify_sems;
}
static int Invar_setModifySems(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        Py_XDECREF(self->modify_sems);
        self->modify_sems = NULL;
        return 0;
    }

    if(!(PyAnySet_Check(value)))
    {
        PyErr_SetString(PyExc_TypeError, "Modify_semaphore objects should be set object");
        return -1;
    }
    tmp = self->modify_sems;
    Py_INCREF(value);
    self->modify_sems = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getSelectQuery(Invar *self, void *closure)/*{{{*/
{
    if(self->select_query == NULL)
    {
        Py_RETURN_NONE;
    }   
    Py_INCREF(self->select_query);
    return self->select_query;
}
static int Invar_setSelectQuery(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->select_query;
    Py_XINCREF(value);
    self->select_query = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getCache(Invar *self, void *closure)/*{{{*/
{
    if(self->cache == NULL)
    {
        self->cache = PyDict_New();
    }   
    Py_XINCREF(self->cache);
    return self->cache;
}
static int Invar_setCache(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cache field may not be deleted");
        return -1;
    }
    if(!PyDict_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Cache field should contain dictionary");
        return -1;
    }
    tmp = self->cache;
    Py_INCREF(value);
    self->cache = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getLocal(Invar *self, void *closure)/*{{{*/
{
    if(self->local == NULL)
    {
        self->local = PyDict_New();
    }   
    Py_XINCREF(self->local);
    return self->local;
}
static int Invar_setLocal(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Local field may not be deleted");
        return -1;
    }
    if(!PyDict_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Local field should contain dictionary");
        return -1;
    }
    tmp = self->local;
    Py_INCREF(value);
    self->local = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Invar_getCapClass(Invar *self, void *closure)/*{{{*/
{
    if(self->cap_class == NULL)
    {
        Py_RETURN_NONE;
    }   
    Py_INCREF(self->cap_class);
    return self->cap_class;
}

static int Invar_setCapClass(Invar *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        Py_XDECREF(self->cap_class);
        self->cap_class = NULL;
        return 0;
    }
    if(!PyType_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Capability class field should be a type");
        return -1;
    }
    tmp = self->cap_class;
    Py_INCREF(value);
    self->cap_class = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyGetSetDef Invar_getseters[] = {
    {"descriptors", 
     (getter)Invar_getDescriptors, (setter)Invar_setDescriptors,
     "Descriptor objects for this container object",
     NULL},
    {"actidx_descriptors", 
     (getter)Invar_getActIdxDescriptors, (setter)Invar_setActIdxDescriptors,
     "Active descriptor indexes",
     NULL},
    {"fields", 
     (getter)Invar_getFields, (setter)Invar_setFields,
     "Field object for this container object",
     NULL},
    {"actidx", 
     (getter)Invar_getActIdx, (setter)Invar_setActIdx,
     "Active field indexes",
     NULL},
    {"dims", 
     (getter)Invar_getDims, (setter)Invar_setDims,
     "Dimensions for this container object",
     NULL},
    {"actidx_dims", 
     (getter)Invar_getActIdxDims, (setter)Invar_setActIdxDims,
     "Active dimension indexes",
     NULL},
    {"segments", 
     (getter)Invar_getSegments, (setter)Invar_setSegments,
     "Segments",
     NULL},
    {"modify_sems", 
     (getter)Invar_getModifySems, (setter)Invar_setModifySems,
     "Modification semaphore set",
     NULL},
    {"select_query", 
     (getter)Invar_getSelectQuery, (setter)Invar_setSelectQuery,
     "Select query specified by this invariant",
     NULL},
    {"cache", 
     (getter)Invar_getCache, (setter)Invar_setCache,
     "Cache for invariants",
     NULL},
    {"local", 
     (getter)Invar_getLocal, (setter)Invar_setLocal,
     "local dictionary for invariants",
     NULL},
    {"cap_class", 
     (getter)Invar_getCapClass, (setter)Invar_setCapClass,
     "Stores capability class for variable part",
     NULL},
    {NULL}  /* Sentinel */
};


static PyObject * Invar_register(Invar *self,PyObject * args)
{
    PyObject *basecon, *key, *res;
    RegisterContext *rc;
    Invar *new_invar=NULL;
    PyObject *tmp;
    long i,args_size;
    args_size = PyTuple_GET_SIZE(args);

    if(args_size == 0)
    {
        PyErr_SetString(PyExc_TypeError, "First argument of register should be source object.");
        return NULL;
    }
    basecon = PyTuple_GET_ITEM(args,0);
    if(!PyObject_IsInstance(basecon,(PyObject *)&BaseContainer_Type))
    {
        PyErr_SetString(PyExc_TypeError, "First argument of register should be source object.");
        return NULL;
    }

    Py_INCREF(basecon->ob_type); //used in key, will be decrefd later
    if(args_size == 1)
        key = (PyObject *) basecon->ob_type;
    else
    {
       key = PyTuple_New(args_size);
       PyTuple_SET_ITEM(key,0,(PyObject *) basecon->ob_type);
       if(key == NULL)
            return NULL;
       for(i = 1; i < args_size; i++)
       {
            tmp = PyTuple_GET_ITEM(args,i);
            Py_INCREF(tmp);
            PyTuple_SET_ITEM(key,i,tmp);
       }
    }

    if(self->cache != NULL)
    {
        new_invar = (Invar *) PyDict_GetItem(self->cache,key);
        Py_XINCREF(new_invar);
    }
    else
    {
        self->cache = PyDict_New();
        if(self->cache == NULL)
        {
            Py_DECREF(key);
            return NULL;
        }
    }

    if(new_invar == NULL)
    {
        new_invar = (Invar *)  Invar_Type.tp_alloc(&Invar_Type,0);
        if(PyDict_SetItem(self->cache,key,(PyObject *) new_invar)==-1)
        {
            Py_XDECREF(new_invar);
            Py_DECREF(key);
            return NULL;
        }
        res = Py_True;
    }
    else
    {

        if(((Invar *)new_invar)->cap_class != NULL)
        {
            //lets do it the safe way so that we check for class conflicts...
            if(PyObject_SetAttrString(basecon,"__class__",((Invar *)new_invar)->cap_class)==-1)
            {
                Py_DECREF(new_invar);
                Py_DECREF(key);
                return NULL;
            }
        }
        res = Py_False;
    }
    tmp = (PyObject *)((BaseContainer *)basecon)->invar;
    ((BaseContainer *)basecon)->invar = new_invar;
    Py_XDECREF(tmp);

    rc = (RegisterContext *) RegisterContext_Type.tp_alloc(&RegisterContext_Type,0);
    if(rc == NULL)
        return NULL;
   
    Py_INCREF(res);
    rc->init = res;

    Py_INCREF(self);
    rc->source_invar = (PyObject *) self;
    
    rc->key = key; //steal reference
    return (PyObject *) rc;
}

static PyMethodDef Invar_methods[] = {
    {"register", (PyCFunction)Invar_register, METH_VARARGS,
     "Register basecontainer with invariant"
    },
    {NULL} 
};


static PyTypeObject Invar_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "base_container.Invar",             /*tp_name*/
    sizeof(Invar),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Invar_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,          /*tp_call*/
    0,                         /*tp_str*/
    0,                          /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_CLASS, /*tp_flags*/
    "Invar object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Invar_methods,         /* tp_methods */
    0,                         /* tp_members */
    Invar_getseters,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    offsetof(Invar,local),      /* tp_dictoffset */
    0,                  /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};/*}}}*/

static void RegisterContext_dealloc(RegisterContext* self)/*{{{*/
{
    Py_XDECREF(self->init);
    Py_XDECREF(self->source_invar);
    Py_XDECREF(self->key);

    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
RegisterContext_enter(RegisterContext *self)
{
    Py_XINCREF(self->init);
    return self->init;
}

static PyObject *
RegisterContext_exit(RegisterContext *self,PyObject *args)
{
    PyObject * cache;
    if(args != NULL && PyTuple_Check(args) && PyTuple_GET_SIZE(args) > 0)
    {
        if(PyTuple_GET_ITEM(args,0) != Py_None)
        {
            if(self->source_invar == NULL || self->key == NULL)
            {
                PyErr_SetString(PyExc_TypeError, "RegisterContext not initialized by invar object.");
                return NULL;
            }
            cache = ((Invar *) self->source_invar)->cache;
            if(cache == NULL)
            {
                PyErr_SetString(PyExc_TypeError, "RegisterContext not initialized by invar object.");
                return NULL;
            }
            
            if(PyDict_DelItem(cache,self->key)==-1)
                return NULL;
        }
    }
    
    Py_INCREF(Py_False);
    return Py_False;
}

static PyMethodDef RegisterContext_methods[] = {
    {"__enter__", (PyCFunction)RegisterContext_enter, METH_NOARGS,
     "Enter context"
    },
    {"__exit__", (PyCFunction)RegisterContext_exit, METH_VARARGS,
     "Exit context"
    },
    {NULL} 
};

static PyTypeObject RegisterContext_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "base_container.RegisterContext",             /*tp_name*/
    sizeof(RegisterContext),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)RegisterContext_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,          /*tp_call*/
    0,                         /*tp_str*/
    0,                          /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_CLASS, /*tp_flags*/
    "RegisterContext object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    RegisterContext_methods,         /* tp_methods */
    0,                         /* tp_members */
    0,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                  /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};/*}}}*/

static void BaseContainer_dealloc(BaseContainer* self)/*{{{*/
{
    Py_XDECREF(self->invar);
    Py_XDECREF(self->result);
    Py_XDECREF(self->source);

    self->ob_type->tp_free((PyObject*)self);
}

static PyObject * BaseContainer_getInvar(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "BaseContainer object not properly initialized (no invar).");
        return NULL;
    }   
    Py_INCREF(self->invar);
    return (PyObject *) self->invar;
}
static int BaseContainer_setInvar(BaseContainer *self, PyObject *value, void *closure)
{
    Invar *tmp;
    if(value == NULL)
    {
        Py_XDECREF(self->invar);
        self->invar = NULL;
        return 0;
    }
    
    if(!(value->ob_type == &Invar_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Invar parameter should be an invar object");
        return -1;
    }
    tmp = self->invar;
    Py_INCREF(value);
    self->invar = (Invar *)value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * BaseContainer_getResult(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->result == NULL)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }   
    Py_INCREF(self->result);
    return self->result;
}
static int BaseContainer_setResult(BaseContainer *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->result;
    Py_XINCREF(value);
    self->result = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * BaseContainer_getSource(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->source == NULL)
    {
        Py_RETURN_NONE;
    }   
    Py_INCREF(self->source);
    return self->source;
}
static int BaseContainer_setSource(BaseContainer *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL)
    {
        Py_XDECREF(self->source);
        self->source = NULL;
        return 0;
    }
    if(!(PyTuple_Check(value)))
    {
        PyErr_SetString(PyExc_TypeError, "Source should be an tuple");
        return -1;
    }
    tmp = self->source;
    Py_INCREF(value);
    self->source = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * BaseContainer_getDescriptors(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getDescriptors(self->invar,closure);
}
static int BaseContainer_setDescriptors(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setDescriptors(self->invar,value,closure);
}/*}}}*/

static PyObject * BaseContainer_getActIdxDescriptors(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getActIdxDescriptors(self->invar,closure);
}
static int BaseContainer_setActIdxDescriptors(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setActIdxDescriptors(self->invar,value,closure);
}/*}}}*/

static PyObject * BaseContainer_getFields(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getFields(self->invar,closure);
}
static int BaseContainer_setFields(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setFields(self->invar,value,closure);
}/*}}}*/

static PyObject * BaseContainer_getActIdx(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getActIdx(self->invar,closure);
}
static int BaseContainer_setActIdx(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setActIdx(self->invar,value,closure);
}/*}}}*/

static PyObject * BaseContainer_getDims(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getDims(self->invar,closure);
}
static int BaseContainer_setDims(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setDims(self->invar,value,closure);
}/*}}}*/

static PyObject * BaseContainer_getActIdxDims(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getActIdxDims(self->invar,closure);
}
static int BaseContainer_setActIdxDims(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setActIdxDims(self->invar,value,closure);
}/*}}}*/

static PyObject * BaseContainer_getSegments(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getSegments(self->invar,closure);
}
static int BaseContainer_setSegments(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setSegments(self->invar,value,closure);
}/*}}}*/

static PyObject * BaseContainer_getModifySems(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getModifySems(self->invar,closure);
}
static int BaseContainer_setModifySems(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setModifySems(self->invar,value,closure);
}/*}}}*/

static PyObject * BaseContainer_getSelectQuery(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getSelectQuery(self->invar,closure);
}
static int BaseContainer_setSelectQuery(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setSelectQuery(self->invar,value,closure);
}/*}}}*/

static PyObject * BaseContainer_getLocal(BaseContainer *self, void *closure)/*{{{*/
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return NULL;
    }   
    return Invar_getLocal(self->invar,closure);
}
static int BaseContainer_setLocal(BaseContainer *self, PyObject *value, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Invar not set!");
        return -1;
    }  
    return Invar_setLocal(self->invar,value,closure);
}/*}}}*/

static PyGetSetDef BaseContainer_getseters[] = {
    {"_invar", 
     (getter)BaseContainer_getInvar, (setter)BaseContainer_setInvar,
     "Invariant part of this container",
     NULL},
    {"_result", 
     (getter)BaseContainer_getResult, (setter)BaseContainer_setResult,
     "Result of query",
     NULL},
    {"_source", 
     (getter)BaseContainer_getSource, (setter)BaseContainer_setSource,
     "Tuple with source basecontainers",
     NULL},
    {"_descriptors", 
     (getter)BaseContainer_getDescriptors, (setter)BaseContainer_setDescriptors,
     "Refers to invar descriptors",
     NULL},
    {"_actidx_descriptors", 
     (getter)BaseContainer_getActIdxDescriptors, (setter)BaseContainer_setActIdxDescriptors,
     "Refers to invar active descriptor index",
     NULL},
    {"_fields", 
     (getter)BaseContainer_getFields, (setter)BaseContainer_setFields,
     "Refers to invar fields",
     NULL},
    {"_actidx", 
     (getter)BaseContainer_getActIdx, (setter)BaseContainer_setActIdx,
     "Refers to invar active field index",
     NULL},
    {"_dims", 
     (getter)BaseContainer_getDims, (setter)BaseContainer_setDims,
     "Refers to invar dims",
     NULL},
    {"_actidx_dims", 
     (getter)BaseContainer_getActIdxDims, (setter)BaseContainer_setActIdxDims,
     "Refers to invar active dim indexes",
     NULL},
    {"_segments", 
     (getter)BaseContainer_getSegments, (setter)BaseContainer_setSegments,
     "Refers to invar segments",
     NULL},
    {"_modify_sems", 
     (getter)BaseContainer_getModifySems, (setter)BaseContainer_setModifySems,
     "Refers to invar modify_sems",
     NULL},
    {"_select_query", 
     (getter)BaseContainer_getSelectQuery, (setter)BaseContainer_setSelectQuery,
     "Refers to invar select query",
     NULL},
    {"_local", 
     (getter)BaseContainer_getLocal, (setter)BaseContainer_setLocal,
     "Local invariant dictionary ",
     NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject BaseContainer_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "base_container.BaseContainer",             /*tp_name*/
    sizeof(BaseContainer),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)BaseContainer_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,          /*tp_call*/
    0,                         /*tp_str*/
    0,                          /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_CLASS, /*tp_flags*/
    "BaseContainer object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,         /* tp_methods */
    0,                         /* tp_members */
    BaseContainer_getseters,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                  /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};/*}}}*/


static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initbase_container(void) 
{
    PyObject* m, *c_api_object;
    static void *PyBContainer_API[PyBContainer_API_pointers];
    
    BaseContainer_Type.tp_new = PyType_GenericNew;
    QueryResult_Type.tp_new = PyType_GenericNew;
    ModificationSemaphore_Type.tp_new = PyType_GenericNew;
    Invar_Type.tp_new = PyType_GenericNew;

    if (PyType_Ready(&ModificationSemaphore_Type) < 0)
        return;
    if (PyType_Ready(&QueryResult_Type) < 0)
        return;
    if (PyType_Ready(&BaseContainer_Type) < 0)
        return;
    if (PyType_Ready(&Invar_Type) < 0)
        return;
    if (PyType_Ready(&RegisterContext_Type) < 0)
        return;


    m = Py_InitModule3("base_container", module_methods,
                       "Base class for containers");

    if (m == NULL)
      return;
    import_array();

    Py_INCREF(&ModificationSemaphore_Type);
    Py_INCREF(&QueryResult_Type);
    Py_INCREF(&BaseContainer_Type);
    Py_INCREF(&Invar_Type);
    PyModule_AddObject(m, "ModificationSemaphore", (PyObject *)&ModificationSemaphore_Type);
    PyModule_AddObject(m, "QueryResult", (PyObject *)&QueryResult_Type);
    PyModule_AddObject(m, "BaseContainer", (PyObject *)&BaseContainer_Type);
    PyModule_AddObject(m, "Invar", (PyObject *)&Invar_Type);

    PyBContainer_API[QueryResult_copy_NUM] = (void *)QueryResult_copy;
     /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)PyBContainer_API, NULL);
    
    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);

}
