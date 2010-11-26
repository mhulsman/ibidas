#include <Python.h>
#include "structmember.h"
#include <string.h>
#include "numpy/arrayobject.h"

static PyTypeObject ModificationSemaphore_Type;/*{{{*/

typedef struct {
    PyObject_HEAD
    long nrmod;
    long curid;
} ModificationSemaphore;

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
    "query_graph.ModificationSemaphore",             /*tp_name*/
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


static PyTypeObject QueryResult_Type;/*{{{*/

typedef struct {
    PyObject_HEAD
    PyObject * data;
    PyObject * nrow;
    PyObject * ncol;
    PyObject * modify_check;
    long sem_state;
    char cacheable; //should be set to true if there is no chance that this result set will change (in any of the dependent querys)
} QueryResult;

static void
QueryResult_dealloc(QueryResult* self)
{
    Py_XDECREF(self->data);
    Py_XDECREF(self->nrow);
    Py_XDECREF(self->ncol);
    Py_XDECREF(self->modify_check);
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
QueryResult_repr(QueryResult* self)
{
    PyObject *nrow,*ncol;
    char * cnrow,*cncol;
    if(self->nrow != NULL && self->ncol != NULL)
    {
        nrow = PyObject_Repr(self->nrow);
        ncol = PyObject_Repr(self->ncol);
        
        if(nrow != NULL && ncol != NULL)
        {
            cnrow = PyString_AsString(nrow);
            cncol = PyString_AsString(ncol);
            return PyString_FromFormat("QR[%s,%s]",cnrow,cncol); 
        }
    }
    else
        PyErr_SetString(PyExc_TypeError, "QueryResult object has no valid nrow or ncol.");
    return NULL;
}

static int
QueryResult_init(QueryResult *self, PyObject *args, PyObject *kwds)
{
   PyObject *data,*nrow,*ncol,*tmp;
   

   static char *kwlist[] = {"data","nrow","ncol",NULL};
   if(!PyArg_ParseTupleAndKeywords(args,kwds,"O!O!O!",kwlist,
            &PyTuple_Type,&data,&PyInt_Type,&nrow,&PyInt_Type,&ncol))
      return -1;

   tmp = self->data;
   Py_INCREF(data);
   self->data = data;
   Py_XDECREF(tmp);
   
   tmp = self->nrow;
   Py_INCREF(nrow);
   self->nrow = nrow;
   Py_XDECREF(tmp);

   tmp = self->ncol;
   Py_INCREF(ncol);
   self->ncol = ncol;
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
   res->nrow = self->nrow;
   res->ncol = self->ncol;
   res->modify_check = self->modify_check;
   res->sem_state = self->sem_state;
   res->cacheable = self->cacheable;

   Py_XINCREF(res->data);
   Py_XINCREF(res->nrow);
   Py_XINCREF(res->ncol);
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
    {"copy", (PyCFunction)QueryResult_copy, METH_NOARGS,
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
    if(!PyTuple_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Data objects should be a tuple");
        return -1;
    }
    tmp = self->data;
    Py_INCREF(value);
    self->data = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
QueryResult_getNRow(QueryResult *self, void *closure)
{
    if(self->nrow == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QueryResult object not initialized.");
        return NULL;
    }   
    Py_INCREF(self->nrow);
    return self->nrow;
}
static int
QueryResult_setNRow(QueryResult *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(!PyInt_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Nrow objects should be a integer");
        return -1;
    }
    tmp = self->nrow;
    Py_INCREF(value);
    self->nrow = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
QueryResult_getNCol(QueryResult *self, void *closure)
{
    if(self->ncol == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QueryResult object not initialized.");
        return NULL;
    }   
    Py_INCREF(self->ncol);
    return self->ncol;
}
static int
QueryResult_setNCol(QueryResult *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(!PyInt_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "NCol objects should be an integer");
        return -1;
    }
    tmp = self->ncol;
    Py_INCREF(value);
    self->ncol = value;
    Py_XDECREF(tmp);
    return 0;
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
    {"nrow", 
     (getter)QueryResult_getNRow, (setter)QueryResult_setNRow,
     "Number of rows",
     NULL},
    {"ncol", 
     (getter)QueryResult_getNCol, (setter)QueryResult_setNCol,
     "Number of columns",
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
    "query_graph.QueryResult",             /*tp_name*/
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


/*{{{*/static PyTypeObject BaseContainer_Type;

typedef struct {
    PyObject_HEAD
    PyObject * props;
    PyObject * actidx;
    PyObject * fields;
    PyObject * invar;
} BaseContainer;

static void
BaseContainer_dealloc(BaseContainer* self)
{
    Py_XDECREF(self->props);
    Py_XDECREF(self->fields);
    Py_XDECREF(self->actidx);
    Py_XDECREF(self->invar);
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
BaseContainer_getProps(BaseContainer *self, void *closure)
{
    if(self->props == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "BaseContainer object not properly initialized (no props).");
        return NULL;
    }   
    Py_INCREF(self->props);
    return self->props;
}
static int
BaseContainer_setProps(BaseContainer *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(!PyArray_Check(value) && ((PyArrayObject*)value)->nd == 2)
    {
        PyErr_SetString(PyExc_TypeError, "Props objects should be a two dimensional numpy object");
        return -1;
    }
    tmp = self->props;
    Py_INCREF(value);
    self->props = value;
    Py_XDECREF(tmp);
    return 0;
}


static PyObject *
BaseContainer_getFields(BaseContainer *self, void *closure)
{
    if(self->fields == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "BaseContainer object not properly initialized (no fields).");
        return NULL;
    }   
    Py_INCREF(self->fields);
    return self->fields;
}
static int
BaseContainer_setFields(BaseContainer *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(!PyArray_Check(value) && ((PyArrayObject*)value)->nd == 2)
    {
        PyErr_SetString(PyExc_TypeError, "Field objects should be a two dimensional numpy object");
        return -1;
    }
    tmp = self->fields;
    Py_INCREF(value);
    self->fields = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
BaseContainer_getActIdx(BaseContainer *self, void *closure)
{
    if(self->actidx == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "BaseContainer object not properly initialized (no actidx).");
        return NULL;
    }   
    Py_INCREF(self->actidx);
    return self->actidx;
}
static int
BaseContainer_setActIdx(BaseContainer *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->actidx;
    Py_INCREF(value);
    self->actidx = value;
    Py_XDECREF(tmp);
    return 0;
}


static PyObject *
BaseContainer_getInvar(BaseContainer *self, void *closure)
{
    if(self->invar == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "BaseContainer object not properly initialized (no invar).");
        return NULL;
    }   
    Py_INCREF(self->invar);
    return self->invar;
}
static int
BaseContainer_setInvar(BaseContainer *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->invar;
    Py_INCREF(value);
    self->invar = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyGetSetDef BaseContainer_getseters[] = {
    {"_props", 
     (getter)BaseContainer_getProps, (setter)BaseContainer_setProps,
     "Props object for this container object",
     NULL},
    {"_fields", 
     (getter)BaseContainer_getFields, (setter)BaseContainer_setFields,
     "Field object for this container object",
     NULL},
    {"_actidx", 
     (getter)BaseContainer_getActIdx, (setter)BaseContainer_setActIdx,
     "Active field indexes",
     NULL},
    {"_invar", 
     (getter)BaseContainer_getInvar, (setter)BaseContainer_setInvar,
     "Invariant part of this container",
     NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject BaseContainer_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "query_graph.BaseContainer",             /*tp_name*/
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


/*{{{*/static PyTypeObject Node_Type;

typedef struct {
    PyObject_HEAD
    PyObject * obj;
    PyObject * source;
    PyObject * target;
    PyObject * fields;
    PyObject * actidx;
    PyObject * props;
} Node;

static void
Node_dealloc(Node* self)
{
    Py_XDECREF(self->obj);
    Py_XDECREF(self->source);
    Py_XDECREF(self->target);
    Py_XDECREF(self->fields);
    Py_XDECREF(self->actidx);
    Py_XDECREF(self->props);
    self->ob_type->tp_free((PyObject*)self);
}


static int
Node_init(Node *self, PyObject *args, PyObject *kwds)
{
   PyObject *obj,*source = NULL,*target = NULL, *fields, *props, *actidx;
   static char *kwlist[] = {"obj","source","target",NULL};
   if(!PyArg_ParseTupleAndKeywords(args,kwds,"O|O!O!",kwlist,
            &obj,&PyList_Type,&source,&PyList_Type,&target))
      return -1;

   if(source == NULL)
       self->source = PyList_New(0);
   else
   {
       Py_INCREF(source);
       self->source = source;
   }
   
   if(target == NULL)
       self->target = PyList_New(0);
   else
   {
       Py_INCREF(target);
       self->target = target;
   }

   if(PyString_Check(obj) || PyInt_Check(obj) || PyType_Check(obj))
   {
        Py_INCREF(obj);
        self->obj = obj;
   }
   else
   {
        Py_INCREF(obj->ob_type);
        self->obj = (PyObject *)obj->ob_type;
   }

   fields = PyObject_GetAttrString(obj,"_fields");
   if(fields != NULL)
   {
       Py_INCREF(fields);
       self->fields = fields;
   }
   else
   {
       Py_INCREF(Py_None);
       self->fields = Py_None;
       PyErr_Clear();
   }
   
   actidx = PyObject_GetAttrString(obj,"_actidx");
   if(actidx != NULL)
   {
       Py_INCREF(actidx);
       self->actidx = actidx;
   }
   else
   {
       Py_INCREF(Py_None);
       self->actidx = Py_None;
       PyErr_Clear();
   }
  
   props = PyObject_GetAttrString(obj,"_props");
   if(props != NULL)
   {
       Py_INCREF(props);
       self->props = props;
   }
   else
   {
       Py_INCREF(Py_None);
       self->props = Py_None;
       PyErr_Clear();
   }
   return 0;
}


static PyObject *
Node_getObj(Node *self, void *closure)
{
    Py_INCREF(self->obj);
    return self->obj;
}
static int
Node_setObj(Node *self, PyObject *value, void *closure)
{
    Py_XDECREF(self->obj);
    Py_INCREF(value);
    self->obj = value;
    return 0;
}


static PyObject *
Node_getSource(Node *self, void *closure)
{
    Py_INCREF(self->source);
    return self->source;
}
static int
Node_setSource(Node *self, PyObject *value, void *closure)
{
    if(!PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Source should be a list object");
        return -1;
    }
    Py_XDECREF(self->source);
    Py_INCREF(value);
    self->source = value;
    return 0;
}

static PyObject *
Node_getTarget(Node *self, void *closure)
{
    Py_INCREF(self->target);
    return self->target;
}
static int
Node_setTarget(Node *self, PyObject *value, void *closure)
{
    if(!PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Target should be a list object");
        return -1;
    }
    Py_XDECREF(self->target);
    Py_INCREF(value);
    self->target = value;
    return 0;
}

static PyObject *
Node_getFields(Node *self, void *closure)
{
    Py_XINCREF(self->fields);
    return self->fields;
}
static int
Node_setFields(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->fields;
    Py_INCREF(value);
    self->fields = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
Node_getActIdx(Node *self, void *closure)
{
    Py_XINCREF(self->actidx);
    return self->actidx;
}
static int
Node_setActIdx(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->actidx;
    Py_INCREF(value);
    self->actidx = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
Node_getProps(Node *self, void *closure)
{
    Py_XINCREF(self->props);
    return self->props;
}
static int
Node_setProps(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->props;
    Py_INCREF(value);
    self->props = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyGetSetDef Node_getseters[] = {
    {"obj", 
     (getter)Node_getObj, (setter)Node_setObj,
     "Object/action a node is wrapped around.",
     NULL},
    {"source", 
     (getter)Node_getSource, (setter)Node_setSource,
     "Source nodes for this node",
     NULL},
    {"target", 
     (getter)Node_getTarget, (setter)Node_setTarget,
     "Target nodes for this node",
     NULL},
    {"fields", 
     (getter)Node_getFields, (setter)Node_setFields,
     "Fields represented by this node",
     NULL},
    {"props", 
     (getter)Node_getProps, (setter)Node_setProps,
     "Properties represented by this node",
     NULL},
    {"actidx", 
     (getter)Node_getActIdx, (setter)Node_setActIdx,
     "Active field indexes",
     NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject Node_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "query_graph.Node",             /*tp_name*/
    sizeof(Node),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Node_dealloc, /*tp_dealloc*/
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
    "Node object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,         /* tp_methods */
    0,                         /* tp_members */
    Node_getseters,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Node_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};
/*}}}*/


/*{{{*/static PyTypeObject MultiVisitorType;

typedef struct {
    PyObject_HEAD
    PyObject * vcache;
    PyObject * prefixes;
} MultiVisitor;

static void
MultiVisitor_dealloc(MultiVisitor* self)
{
    Py_XDECREF(self->vcache);
    Py_XDECREF(self->prefixes);
    self->ob_type->tp_free((PyObject*)self);
}

static int
MultiVisitor_init(MultiVisitor *self, PyObject *args, PyObject *kwds)
{
    PyObject *key, *vcache, *prefixes = NULL, *nprefixes, *tmp, *name, *func;
    int i;
    long prefix_size;

    static char *kwlist[] = {"prefixes",NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O!", kwlist, (PyObject *)&PyTuple_Type, &prefixes)) 
        return -1;

    if(prefixes == NULL)
        nprefixes = PyTuple_New(1);
    else
        nprefixes = PyTuple_New(PyTuple_GET_SIZE(prefixes)+1);
    if(nprefixes == NULL) 
        return -1;
    tmp = PyString_FromString("visit");
    if(tmp == NULL)
        return -1;
    PyTuple_SET_ITEM(nprefixes,0,tmp);

    if(prefixes != NULL)
    {
        prefix_size = PyTuple_GET_SIZE(prefixes);
        for(i = 0; i < prefix_size; i++)
        {
            tmp = PyTuple_GET_ITEM(prefixes,i);
            if(!PyString_Check(tmp))
            {
                Py_DECREF(nprefixes);
                PyErr_SetString(PyExc_TypeError, "Prefixes should be strings.");
                return -1;
            }
            Py_INCREF(tmp);
            PyTuple_SET_ITEM(nprefixes,i + 1,tmp);
            name = PyString_FromFormat("__visit%d",i+1);
            if(name == NULL)
            {
                Py_DECREF(nprefixes);
                return -1;
            }
            func = PyObject_GetAttr((PyObject *)self,name);
            Py_DECREF(name);
            if(func == NULL || !PyCallable_Check(func))
            {
                Py_DECREF(nprefixes);
                Py_XDECREF(func);
                return -1;
            }
            if(PyObject_SetAttr((PyObject *)self,tmp,func) < 0)
            {
                Py_DECREF(nprefixes);
                Py_DECREF(func);
                return -1;
            }
            Py_DECREF(func);

        }
    }
    self->prefixes = nprefixes;
    
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

static PyObject * MultiVisitor_visit(MultiVisitor *self,PyObject * prefix, PyObject *args)
{
    Py_ssize_t arglength;
    PyObject * visited;
    PyObject * visit_method;
    PyObject * visit_args;
    PyObject * res, * key;
    int i = 0;


    if(self->vcache == NULL)
         if(MultiVisitor_init(self,NULL,NULL) == -1)
            return 0;
    
    if(!PyTuple_Check(args))
    {
        PyErr_SetString(PyExc_TypeError, "Visit method should be called with the object to visit.");
        return 0;
    }
    if(prefix == NULL || !PyString_Check(prefix))
    {
        PyErr_SetString(PyExc_TypeError, "Visit method have valid string prefix.");
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
    
    
    key = Py_BuildValue("OO",prefix,(PyObject *) visited->ob_type);
    if(key == NULL)
        return NULL;
    visit_method = PyDict_GetItem(self->vcache,key);
    if(visit_method == NULL)
    {
        PyObject * mro = PyObject_CallMethod((PyObject *)visited->ob_type,"mro",NULL);    //visited->ob_type->tp_mro;  #fails for floats
        PyObject * cur_class;
        PyObject * curname;
        const char *s;
        char *sn;
        char *cprefix;
        int prefix_length;
        int name_length;

        if(mro == NULL || !PyList_Check(mro))
        {
            PyErr_SetString(PyExc_TypeError, "MRO error.");
            Py_DECREF(key); 
            Py_XDECREF(mro);
            return 0;
        }
        
        sn = PyMem_Malloc(sizeof(char) * 1024);
        if(sn == NULL)
        {
            Py_DECREF(key); 
            Py_DECREF(mro);
            return 0;
        }

        cprefix = PyString_AsString(prefix);
        prefix_length = strlen(cprefix);
        Py_MEMCPY(sn,cprefix,prefix_length + 1);  //also copy /0 character to be sure

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

            if(name_length > 1023 - prefix_length)
            {
                PyErr_SetString(PyExc_TypeError, "Visit class name too long.");
                PyMem_Free(sn);
                Py_DECREF(key); 
                return 0;
            }
            Py_MEMCPY(&sn[prefix_length],s,name_length + 1);
            curname = PyString_FromString(sn);
            if(curname == NULL)
            {
                PyMem_Free(sn);
                Py_DECREF(key); 
                return 0;
            }

            if(PyObject_HasAttr((PyObject *) self->ob_type,curname))
            {
                visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
                Py_DECREF(curname);
                if(visit_method == NULL ||
                    PyDict_SetItem(self->vcache,key,visit_method) == -1)
                {
                    Py_XDECREF(visit_method);
                    Py_DECREF(key); 
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
            res = PyObject_Repr((PyObject *)visited);
            if(res == NULL)
                PyErr_SetString(PyExc_TypeError, "No suitable visit method found.");
            else
            {
                PyErr_Format(PyExc_TypeError, "No suitable visit method found for %s,%s.",PyString_AsString(prefix),PyString_AsString(res));
                Py_DECREF(res);
            }
            Py_DECREF(key); 
            return 0;
        }
    }
    else
        Py_INCREF(visit_method);
   
    Py_DECREF(key); 
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

#define MV_VISIT(functionnr) \
static PyObject * MultiVisitor_visit##functionnr(MultiVisitor *self,PyObject *args) \
{ \
   if(self->prefixes == NULL || !PyTuple_Check(self->prefixes) || PyTuple_GET_SIZE(self->prefixes) < (functionnr)) \
   { \
        PyErr_SetString(PyExc_TypeError, "Visitor not correctly initialized (no prefixes)."); \
        return NULL; \
   } \
   return MultiVisitor_visit(self,PyTuple_GET_ITEM(self->prefixes,functionnr),args); \
}

MV_VISIT(0);
MV_VISIT(1);
MV_VISIT(2);
MV_VISIT(3);
MV_VISIT(4);
MV_VISIT(5);
MV_VISIT(6);
MV_VISIT(7);
MV_VISIT(8);

#define MV_TABLE_ENTRY(functionnr) \
    {"__visit" #functionnr, (PyCFunction)MultiVisitor_visit##functionnr, METH_VARARGS, \
     "Visit class by calling appropiate method" \
    }


/* Currently max 8 extra visit methods */
static PyMethodDef MultiVisitor_methods[] = {
    {"visit", (PyCFunction)MultiVisitor_visit0, METH_VARARGS,
     "Visit class by calling appropiate method"
    },
    MV_TABLE_ENTRY(1),
    MV_TABLE_ENTRY(2),
    MV_TABLE_ENTRY(3),
    MV_TABLE_ENTRY(4),
    MV_TABLE_ENTRY(5),
    MV_TABLE_ENTRY(6),
    MV_TABLE_ENTRY(7),
    MV_TABLE_ENTRY(8),
    {NULL} 
};


static PyTypeObject MultiVisitorType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "query_graph.MultiVisitor",             /*tp_name*/
    sizeof(MultiVisitor),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MultiVisitor_dealloc, /*tp_dealloc*/
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
    "MultiVisitor object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    MultiVisitor_methods,         /* tp_methods */
    0,                         /* tp_members */
    0,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MultiVisitor_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};/*}}}*/


/*{{{*/static PyTypeObject QGCreator_Type;

#define NO_MODIFICATION_CHECK 1
#define MODIFICATION_CHECK 2
#define NO_MODIFIABLES 3

#define NO_QG 0
#define QG_DONE 1


typedef struct {
    MultiVisitor v;
    PyObject * root;
    PyObject * cache;
    PyObject * params;
    PyObject * objs;
    PyObject * qg_node;
    
    PyObject * modifiables;
    PyObject * cur_sems;

    long cur_param_pos;
    long max_param_length;
} QGCreator;

static void
QGCreator_dealloc(QGCreator* self)
{
    Py_XDECREF(self->root);
    Py_XDECREF(self->cache);
    Py_XDECREF(self->params);
    Py_XDECREF(self->objs);
    Py_XDECREF(self->qg_node);
    
    Py_XDECREF(self->modifiables);
    Py_XDECREF(self->cur_sems);
    Py_XDECREF(((MultiVisitor *)self)->vcache);
    Py_XDECREF(((MultiVisitor *)self)->prefixes);
    ((MultiVisitor *)self)->ob_type->tp_free((PyObject*)self);
}

static int QGCreator_init(QGCreator * self, PyObject * args, PyObject *kwds)
{
    PyObject *nargs,*nkwds;
    int res;
    nargs= Py_BuildValue("((s))","param");
    nkwds = PyDict_New();
    if(nargs == NULL || nkwds == NULL)
        return -1;
    
    res = MultiVisitor_init((MultiVisitor *)self,nargs,nkwds);
    Py_DECREF(nargs);
    Py_DECREF(nkwds);

    return res;

}

static PyObject * QGCreator_visit(QGCreator * self, PyObject * args)
{
    PyObject *visited, *source,*target;
    PyObject *res, *item, *rootdict, *tmp;
    Py_ssize_t ssize;
    int i;
    if(args == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Visit function called with NULL argument.");
        return NULL;
    }

    if(PyTuple_Check(args))
    {
        if(PyTuple_GET_SIZE(args) == 0)
        {
            PyErr_SetString(PyExc_TypeError, "Incorrect number of arguments given to visit function.");
            return NULL;
        }
        visited = PyTuple_GET_ITEM(args,0);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect argument type given to visit function.");
        return NULL; 
    }

    if(PyObject_IsInstance(visited,(PyObject *) &BaseContainer_Type))
    {
        if(self->cache == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "QGCreator has not been properly initialized.");
            return NULL;
        }
        res = PyDict_GetItem(self->cache,visited);
        if(res == NULL)
        {
            if(self->objs == NULL)
            {
                PyErr_SetString(PyExc_TypeError, "QGCreator has not been properly initialized (objs == NULL).");
                return NULL;
            }
            if(PySet_Add(self->objs,(PyObject *)visited->ob_type) < 0)
                return NULL;

            res = MultiVisitor_visit0((MultiVisitor *) self,args);
            if(res == NULL)
                return NULL;
            
            if(res->ob_type == &Node_Type)
            {
                source = ((Node *) res)->source;
                if(source == NULL || !PyList_Check(source))
                {
                    PyErr_SetString(PyExc_TypeError, "Node object has incorrect source attribute.");
                    return NULL;
                }
                ssize = PyList_GET_SIZE(source);
                for(i = 0; i < ssize; i++)
                {
                    item = PyList_GET_ITEM(source,i);
                    if(item->ob_type == &Node_Type)
                    {
                        target = ((Node *)item)->target;
                        if(target == NULL || !PyList_Check(target))
                        {
                            PyErr_SetString(PyExc_TypeError, "Node object has incorrect target attribute.");
                            return NULL;
                        }
                        PyList_Append(target,res);
                    }
                }
            }
            
            rootdict = PyObject_GetAttrString(visited,"__dict__");
            if(rootdict == NULL || !PyDict_Check(rootdict))
            {
                PyErr_SetString(PyExc_TypeError, "Could not obtain dictionary object of root object.");
                return NULL;
            }
            tmp = PyDict_GetItemString(rootdict,"modify_sems");
            if(tmp != NULL)
            {
                PyList_Append(self->modifiables,visited);
                if(self->cur_sems == NULL)
                    self->cur_sems = PySet_New(tmp);
                else
                    PyNumber_InPlaceAnd(self->cur_sems,tmp);
            }
            Py_DECREF(rootdict);

            
            if(PyDict_SetItem(self->cache,visited,res) < 0)
                return NULL;
        }
        else
            Py_INCREF(res);
        
        return res;
    }
    else
    {
        if(self->params == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "QGCreator has not been properly initialized.");
            return NULL;
        }
        PyList_Append(self->params,visited);
        return PyInt_FromLong(PyList_GET_SIZE(self->params) - 1);
    }
}


static PyObject * QGCreator_param(QGCreator * self, PyObject * args)
{
    PyObject *visited;
    PyObject *res;
    
    if(args == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Visit function called with NULL argument.");
        return NULL;
    }

    if(PyTuple_Check(args))
    {
        if(PyTuple_GET_SIZE(args) == 0)
        {
            PyErr_SetString(PyExc_TypeError, "Incorrect number of arguments given to visit function.");
            return NULL;
        }
        visited = PyTuple_GET_ITEM(args,0);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect argument type given to visit function.");
        return NULL; 
    }

    if(PyObject_IsInstance(visited,(PyObject *) &BaseContainer_Type))
    {
        if(self->cache == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "QGCreator has not been properly initialized.");
            return NULL;
        }
        res = PyDict_GetItem(self->cache,visited);
        if(res == NULL)
        {
            res = MultiVisitor_visit1((MultiVisitor *) self,args);
            if(res == NULL)
                return NULL;

            if(PyDict_SetItem(self->cache,visited,res) < 0)
                return NULL;
        }
        else
            Py_INCREF(res);
        
        return res;

    }
    else
    {
        if(self->params == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "QGCreator has not been properly initialized.");
            return NULL;
        }
        if(self->cur_param_pos == self->max_param_length)
        {
            PyErr_SetString(PyExc_TypeError, "Param vector has changed in length (bug)");
            return NULL;
        }
        Py_INCREF(visited);
        PyList_SET_ITEM(self->params,self->cur_param_pos,visited);
        return PyInt_FromLong(self->cur_param_pos++);
    }
}

static PyObject * QGCreator_execute(QGCreator *self,PyObject *root)
{
    PyObject *tmp,*qg,*args;
    if(!PyObject_IsInstance(root,(PyObject *) &BaseContainer_Type))
    {
        PyErr_SetString(PyExc_TypeError, "First parameter should be a BaseContainer object");
        return NULL;
    }
   
    /* reset cache */
    tmp = self->cache;
    self->cache = PyDict_New();
    Py_XDECREF(tmp);
    
    /* reset parameters */
    tmp = self->params;
    self->params = PyList_New(0);
    Py_XDECREF(tmp);

    tmp = self->objs;
    self->objs = PySet_New(NULL);
    Py_XDECREF(tmp);
    
    if(self->params == NULL || self->cache == NULL || self->objs == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QGCreator object not properly initialized.");
        return NULL;
    }

    tmp = self->root;
    Py_INCREF(root);
    self->root = root;
    Py_XDECREF(tmp);

    /* reset modifiable state */
    tmp = self->modifiables;
    self->modifiables = PyList_New(0);
    Py_XDECREF(tmp);
    Py_XDECREF(self->cur_sems);
    self->cur_sems = NULL;

    
    args = Py_BuildValue("(O)",root);
    if(args == NULL)
        return NULL;
    qg = QGCreator_visit(self,args);
    Py_DECREF(args);
    if(qg == NULL)
        return NULL;
    
    
    tmp = self->qg_node;
    self->qg_node = qg;  //steal reference
    Py_XDECREF(tmp);

    return Py_BuildValue("OOO",self->qg_node,self->params,self->objs);
}

static PyObject * QGCreator_update(QGCreator *self,PyObject *root)
{
    PyObject *tmp,*args, *res;
    if(!PyObject_IsInstance(root,(PyObject *) &BaseContainer_Type))
    {
        PyErr_SetString(PyExc_TypeError, "First parameter should be a BaseContainer object");
        return NULL;
    }
   
    /* reset cache */
    tmp = self->cache;
    self->cache = PyDict_New();
    Py_XDECREF(tmp);
    
    /* reset parameters */
    tmp = self->params;
    if(tmp != NULL && PyList_Check(tmp))
        self->max_param_length = PyList_GET_SIZE(tmp);
    else
        self->max_param_length = 0;;
    self->params = PyList_New(self->max_param_length);
    self->cur_param_pos = 0;
    Py_XDECREF(tmp);

    if(self->params == NULL || self->cache == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QGCreator object not properly initialized.");
        return NULL;
    }

    /* update root */
    tmp = self->root;
    Py_INCREF(root);
    self->root = root;
    Py_XDECREF(tmp);

    args = Py_BuildValue("(O)",root);
    if(args == NULL)
        return NULL;
    res = QGCreator_param(self,args);

    if(res == NULL)
    {
        Py_DECREF(args);
        return NULL;
    }
    Py_DECREF(res);
    Py_DECREF(args);

    if(self->cur_param_pos != self->max_param_length)
    {
        PyErr_SetString(PyExc_TypeError, "Param vector length has changed!");
        return NULL;
    }

    Py_INCREF(self->params);
    return self->params;
}

static PyObject * QGCreator_paramMultiCon(QGCreator *self,PyObject *args)
{
    PyObject *visited, *source, *source_fast, * item, *nargs, *res;
    int nsource, i;

    if(PyTuple_Check(args))
    {
        if(PyTuple_GET_SIZE(args) == 0)
        {
            PyErr_SetString(PyExc_TypeError, "Incorrect number of arguments given to visit function.");
            return NULL;
        }
        visited = PyTuple_GET_ITEM(args,0);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect argument type given to visit function.");
        return NULL; 
    }

    if(!PyObject_IsInstance(visited,(PyObject *) &BaseContainer_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect argument type given to visit function.");
        return NULL;
    }

    source = PyObject_GetAttrString(visited,"_source");
    if(source == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "MultiCon container has no _source parameter.");
        Py_XDECREF(source);
        return NULL;
    }
    source_fast = PySequence_Fast(source,"source attribute is not a sequence");
    Py_DECREF(source);
    if(source_fast == NULL)
        return NULL;
    nsource = PySequence_Fast_GET_SIZE(source_fast);
    
    for(i = 0; i < nsource; i++)
    {
        item = PySequence_Fast_GET_ITEM(source_fast,i);
        nargs = Py_BuildValue("(O)",item);
        if(nargs == NULL)
        {
            Py_DECREF(source_fast);
            return NULL;
        }
        res = QGCreator_param(self,nargs);
        Py_DECREF(nargs);
        if(res == NULL)
        {
            Py_DECREF(source_fast);
            return NULL;
        }
        Py_DECREF(res);
    }
    Py_DECREF(source_fast);
    Py_INCREF(Py_None);
    return Py_None;
}


static PyMethodDef QGCreator_methods[] = {
    {"execute", (PyCFunction)QGCreator_execute, METH_O,
     "Execute query graphc creation (or update)"
    },
    {"update", (PyCFunction)QGCreator_update, METH_O,
     "Update query graph param vector for new root"
    },
    {"visit", (PyCFunction)QGCreator_visit, METH_VARARGS,
     "Overloaded visit function for qg construction"
    },
    {"__visit1", (PyCFunction)QGCreator_param, METH_VARARGS,
     "Overloaded param function for qg construction"
    },
    {"paramMultiOpCon", (PyCFunction)QGCreator_paramMultiCon, METH_VARARGS,
     "Overloaded param function for MultiCon classes."
    },
    {NULL} 
};

static PyObject *
QGCreator_getQGNode(QGCreator *self, void *closure)
{
    if(self->qg_node == NULL)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }
    else
    {
        Py_INCREF(self->qg_node);
        return self->qg_node;
    }
}

static PyObject *
QGCreator_getRoot(QGCreator *self, void *closure)
{
    if(self->root == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Root object has not been set (use execute function)");
        return NULL;
    }   
    Py_INCREF(self->root);
    return self->root;
}

static PyObject *
QGCreator_getModifiables(QGCreator *self, void *closure)
{
    if(self->modifiables == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Modifiable object has not been set(bug).");
        return NULL;
    }
    
    Py_INCREF(self->modifiables);
    return self->modifiables;
}

static PyObject *
QGCreator_getCurSems(QGCreator *self, void *closure)
{
    if(self->cur_sems == NULL)
    {
        Py_INCREF(Py_None);
        return Py_None;
    }  
    else
    {
        Py_INCREF(self->cur_sems);
        return self->cur_sems;
    }
}

static PyGetSetDef QGCreator_getseters[] = {
    {"qg", 
     (getter)QGCreator_getQGNode, NULL,
     "QG root node.",
     NULL},
    {"root", 
     (getter)QGCreator_getRoot, NULL,
     "Root object for the query graph.",
     NULL},
    {"modifiables", 
     (getter)QGCreator_getModifiables, NULL,
     "Obtain modifiable containers in query graph",
     NULL},
    {"_cur_sems", 
     (getter)QGCreator_getCurSems, NULL,
     "Obtain semaphore selection (for internal use).",
     NULL},
    {NULL}  /* Sentinel */
};



static PyTypeObject QGCreator_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "query_graph.QGCreator",             /*tp_name*/
    sizeof(QGCreator),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)QGCreator_dealloc, /*tp_dealloc*/
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
    "QGCreator object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    QGCreator_methods,         /* tp_methods */
    0,                         /* tp_members */
    QGCreator_getseters,                     /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)QGCreator_init,                    /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};
/*}}}*/


static PyTypeObject Node_MultiVisitorType;/*{{{*/

typedef struct {
    PyObject_HEAD
    PyObject * vcache;
    PyObject * params;
    PyObject * prefixes;
} Node_MultiVisitor;

static void
Node_MultiVisitor_dealloc(Node_MultiVisitor* self)
{
    Py_XDECREF(self->vcache);
    Py_XDECREF(self->params);
    Py_XDECREF(self->prefixes);
    self->ob_type->tp_free((PyObject*)self);
}


/* Initialize visitor with prefixes */
static int
Node_MultiVisitor_init(Node_MultiVisitor *self, PyObject *args, PyObject *kwds)
{
    PyObject *key, *vcache, *tmp, *prefixes = NULL, *nprefixes, *name, *func;
    int i;
    long prefix_size;


    static char *kwlist[] = {"prefixes",NULL};
    if(!PyArg_ParseTupleAndKeywords(args,kwds,"|O!",kwlist,
            (PyObject *)&PyTuple_Type,&prefixes))
      return -1;
   
    /* Construct nprefixes tuple */
    if(prefixes == NULL)
        nprefixes = PyTuple_New(1);
    else
        nprefixes = PyTuple_New(PyTuple_GET_SIZE(prefixes)+1);
    if(nprefixes == NULL) 
        return -1;
    
    /* Visit method should always be part of prefixes */
    tmp = PyString_FromString("visit");
    if(tmp == NULL)
    {
        Py_DECREF(nprefixes);
        return -1;
    }
    PyTuple_SET_ITEM(nprefixes,0,tmp);

    
    /* If there are other prefixes, store them also in the list and attach them to a __visit%d method */
    if(prefixes != NULL)
    {
        prefix_size = PyTuple_GET_SIZE(prefixes);
        for(i = 0; i < prefix_size; i++)
        {
            tmp = PyTuple_GET_ITEM(prefixes,i);
            /* Check if prefix is a string */
            if(!PyString_Check(tmp))
            {
                Py_DECREF(nprefixes);
                PyErr_SetString(PyExc_TypeError, "Prefixes should be strings.");
                return -1;
            }
            Py_INCREF(tmp);
            PyTuple_SET_ITEM(nprefixes,i + 1,tmp);

            /* Obtain accompanying visit method */
            name = PyString_FromFormat("__visit%d",i+1);
            if(name == NULL)
            {
                Py_DECREF(nprefixes);
                return -1;
            }
            func = PyObject_GetAttr((PyObject *)self,name);
            Py_DECREF(name);
            if(func == NULL || !PyCallable_Check(func))
            {
                Py_DECREF(nprefixes);
                Py_XDECREF(func);
                return -1;
            }
            
            /* Now store alias from prefix to the __visit%d method */
            if(PyObject_SetAttr((PyObject *)self,tmp,func) < 0)
            {
                Py_DECREF(nprefixes);
                Py_DECREF(func);
                return -1;
            }
            Py_DECREF(func);
             
            /* Do the same of __visitsource%d */
            name = PyString_FromFormat("__visitsource%d",i+1);
            if(name == NULL)
            {
                Py_DECREF(nprefixes);
                return -1;
            }
            func = PyObject_GetAttr((PyObject *)self,name);
            Py_DECREF(name);
            if(func == NULL || !PyCallable_Check(func))
            {
                Py_DECREF(nprefixes);
                Py_XDECREF(func);
                return -1;
            }
            Py_INCREF(tmp); 
            /* Construct prefix + source alias */
            PyString_ConcatAndDel(&tmp,PyString_FromString("source"));
            if(tmp == NULL)
            {
                Py_DECREF(nprefixes);
                Py_DECREF(func);
                return -1;
            }
            /* Store alias to __visitsource%d method */
            if(tmp == NULL || PyObject_SetAttr((PyObject *)self,tmp,func) < 0)
            {
                Py_DECREF(nprefixes);
                Py_DECREF(func);
                Py_DECREF(tmp);
                return -1;
            }
            Py_DECREF(tmp);
            Py_DECREF(func);
        }
    }

    self->prefixes = nprefixes;


    /* Store vcache locally */
    if(self->vcache == NULL)
    {
        /* vcache is stored in class dict as __cache__ */ 
        key = PyString_FromString("__cache__");
        if(key == NULL)
            return -1;

        vcache = PyDict_GetItem(self->ob_type->tp_dict,key);
        /* If no vcache found, create one and store in class dictionary */
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

/* Visitor method for Node objects.*/

static PyObject * Node_MultiVisitor_visit(Node_MultiVisitor *self,PyObject * prefix, PyObject *args)
{
    Py_ssize_t arglength;
    PyObject * visited,*visitid;
    PyObject * visit_method;
    PyObject * visit_args;
    PyObject * res, * key;
    int i = 0;
    char is_param = 0;
    char *sn;

    
    /* vcache stores the functions which was called for a specific object-type 
     * (shared by multiple instances of an specific visitor class) */
    if(self->vcache == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Visitor object has not correctly been initialized (no vcache)");
        return 0;
    }
    
    /* Check if something to visit is in the arguments */
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
   

    /* If the value is an int, it is an index into the param array. */
    if(PyInt_Check(visited))
    {
        /* Check if there is actually a good, long enough, param list available */
        long pidx = PyInt_AsLong(visited);
        if(self->params == NULL || (PyList_GET_SIZE(self->params)-1) < pidx)
        {
            PyErr_SetString(PyExc_TypeError, "Paramlist array has not been (correctly) initialized.");
            return 0;
        }

        /* update object to visit*/
        visited = PyList_GET_ITEM(self->params,pidx);
        if(visited == NULL)
            return NULL;
        
        //QueryResult objects are copied as there version in the query graph tree
        //may not be changed
        if(visited->ob_type == &QueryResult_Type)
            visited = QueryResult_copy((QueryResult *)visited);
        else
            Py_INCREF(visited);
        
        visitid = (PyObject *)visited->ob_type;
        is_param = 1;
    }
    else
    {
        
        if(visited->ob_type != &Node_Type || ((Node *)visited)->obj == NULL)
        {
            //if not a node or a int, then it is just data that we directly return
            //QueryResult objects are copied as there version in the query graph tree
            //may not be changed
            if(visited->ob_type == &QueryResult_Type)
                visited = QueryResult_copy((QueryResult *)visited);
            else
                Py_INCREF(visited);
            visitid = (PyObject *)visited->ob_type;
            is_param = 1;
        } 
        else
        {
            Py_INCREF(visited);
            visitid = ((Node *)visited)->obj;
        }
    }

    /* visitid is the key for the vcache, prefix the prefix for the methods which should be called */
    key = Py_BuildValue("OO",prefix,(PyObject *) visitid);
    if(key == NULL)
        goto VisitError;
    
    /* Lets see if it is available in the vcache */
    visit_method = PyDict_GetItem(self->vcache,key);
    /* If we did not find a method in an earlier search, we store Py_None in vcache */
    if(visit_method == Py_None)
    {
        Py_DECREF(key);
        return visited;
    }

    /* Visit method was not found in cache, have to find it ourself */ 
    if(visit_method == NULL)
    {
        PyObject * cur_class;
        PyObject * curname;
        PyObject * mro;
        const char *s;
        char *cprefix;
        int name_length;
        int prefix_length;

        /* Allocate method name buffer */
        sn = PyMem_Malloc(sizeof(char) * 1024);
        if(sn == NULL)
            goto VisitError2;
        
        /* First part of the buffer is filled with the prefix */
        cprefix = PyString_AsString(prefix);
        prefix_length = strlen(cprefix);
        Py_MEMCPY(sn,cprefix,prefix_length + 1); //also copy /0 character to be sure
       
        /* If visitid was not an string it was another object. 
         * We will walk through its MRO, and see if prefix + mro-classname 
         * is an method in this visitor class */
        if(!PyString_Check(visitid))
        {
            /* Obtain mro for the visitid */
            mro = PyObject_CallMethod((PyObject *)visitid,"mro",NULL);    //visited->ob_type->tp_mro;  #fails for floats
            if(mro == NULL || !PyList_Check(mro))
            {
                PyErr_SetString(PyExc_TypeError, "MRO error.");
                Py_XDECREF(mro);
                goto VisitError3;
            }
            
            /* Walk through MRO */
            Py_ssize_t mrolength = PyList_GET_SIZE(mro);
            for(i = 0; i < mrolength; i++)
            {
                cur_class = PyList_GET_ITEM(mro,i);

                /*Get class name */
                s = strrchr(((PyTypeObject *)cur_class)->tp_name, '.');
                if (s == NULL)
                    s = ((PyTypeObject * )cur_class)->tp_name;
                else
                    s++;
                name_length = strlen(s);
                if(name_length > (1023 - prefix_length))
                {
                    PyErr_SetString(PyExc_TypeError, "Visit class name too long.");
                    Py_DECREF(mro);
                    goto VisitError3;
                }
                
                /* Add class name to method name buffer  */
                Py_MEMCPY(&sn[prefix_length],s,name_length + 1);
                curname = PyString_FromString(sn);
                if(curname == NULL)
                {
                    Py_DECREF(mro);
                    goto VisitError3;
                }

                /* Determine if prefix + class name is a method */
                if(PyObject_HasAttr((PyObject *) self->ob_type,curname))
                {
                    /* If so, obtain method and store it in vcache */
                    visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
                    Py_DECREF(curname);
                    if(visit_method == NULL ||
                        PyDict_SetItem(self->vcache,key,visit_method) == -1)
                    {
                        Py_XDECREF(visit_method);
                        Py_DECREF(mro);
                        goto VisitError3;
                    }
                    break;
                }
                Py_DECREF(curname);
            }
            Py_DECREF(mro);
        }
        else /* visitid Object is an string */
        {
            /* Add visitid to method name buffer */
            s = PyString_AsString(visitid);
            name_length = strlen(s);
            if(name_length > (1023 - prefix_length))
            {
                PyErr_SetString(PyExc_TypeError, "Visit class name too long.");
                goto VisitError3;
            }
            Py_MEMCPY(&sn[prefix_length],s,name_length + 1);
            curname = PyString_FromString(sn);
            if(curname == NULL)
                goto VisitError3;

            /* Check if prefix + visitid string is a method */
            if(PyObject_HasAttr((PyObject *) self->ob_type,curname))
            {
                /* If so, obtain method and store it in vcache */
                visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
                if(visit_method == NULL ||
                    PyDict_SetItem(self->vcache,key,visit_method) == -1)
                {
                    Py_DECREF(curname);
                    Py_XDECREF(visit_method);
                    goto VisitError3;
                }
            }
            Py_DECREF(curname);
        }

        //If no method is found, search for method with name <prefix>Else
        if(visit_method == NULL)
        {
            Py_MEMCPY(&sn[prefix_length],"Else",4 + 1);
            curname = PyString_FromString(sn);
            if(curname == NULL)
                goto VisitError3;

            /* Check if prefix + Else string is a method */
            if(PyObject_HasAttr((PyObject *) self->ob_type,curname))
            {
                /* If so, obtain method and store it in vcache */
                visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
                if(visit_method == NULL ||
                    PyDict_SetItem(self->vcache,key,visit_method) == -1)
                {
                    Py_DECREF(curname);
                    Py_XDECREF(visit_method);
                    goto VisitError3;
                }
            }
            Py_DECREF(curname);
            
        }
        /* Free method name buffer */ 
        PyMem_Free(sn);

        //Still no method found? then if not a node, return value
        //if node, fail
        if(visit_method == NULL)
        {
            if(is_param)
            {
                Py_INCREF(Py_None);
                if(PyDict_SetItem(self->vcache,key,Py_None) == -1)
                    goto VisitError2;
                return visited; 
            }
            else
            {
                res = PyObject_Repr(visitid);
                if(res == NULL)
                    PyErr_SetString(PyExc_TypeError, "No suitable visit method found");
                else
                {
                    PyErr_Format(PyExc_TypeError, "No suitable visit method found for %s,%s.",PyString_AsString(prefix),PyString_AsString(res));
                    Py_DECREF(res);
                }
                goto VisitError2;
            }
        }
    }
    else
        Py_INCREF(visit_method);
   
    Py_DECREF(key); 
   
    /* Make sure visit_method is actually callable */
    if(!PyCallable_Check(visit_method))
    {
        PyErr_SetString(PyExc_TypeError, "Cached visit method is not callable.");
        Py_DECREF(visit_method);
        goto VisitError;
    }
     
    /* Construct arguments */
    visit_args = PyTuple_New(arglength+1);
    if(visit_args == NULL)
    {
        Py_DECREF(visit_method);
        goto VisitError;
    }
    
    Py_INCREF(self);
    PyTuple_SET_ITEM(visit_args,0,(PyObject *)self);
    PyTuple_SET_ITEM(visit_args,1,(PyObject *)visited);
     
    /* Copy extra arguments given to this function */
    for(i = 1; i < arglength; i++)
    {
        PyObject * tmp = PyTuple_GET_ITEM(args,i);
        Py_XINCREF(tmp);
        PyTuple_SET_ITEM(visit_args,i+1,tmp);
    }
    
    /* Call visit method */
    res = PyObject_CallObject(visit_method,visit_args);
    Py_DECREF(visit_method);
    Py_DECREF(visit_args);
    return res;

VisitError3:
    PyMem_Free(sn);
VisitError2:
    Py_DECREF(key);
VisitError:
    Py_DECREF(visited);
    return 0;
}


/* Visit the sources of a node */
static PyObject * Node_MultiVisitor_visitsources(Node_MultiVisitor *self,PyObject *prefix, PyObject *args)
{
    PyObject *visited;
    PyObject *source;
    Py_ssize_t arglength;

    /* Check if there is an argument */
    if(!PyTuple_Check(args))
    {
        PyErr_SetString(PyExc_TypeError, "visitsources method should be called with the object to visit.");
        return 0;
    }
    arglength = PyTuple_GET_SIZE(args);
    if(arglength == 0)
    {
        PyErr_SetString(PyExc_TypeError, "visitsources method should be called with the object to visit.");
        return 0;
    }
   
    /* Get argument  and check if it is a node */
    visited = PyTuple_GET_ITEM(args,0);
    if(visited == NULL)
        return 0;
    if(!(visited->ob_type == &Node_Type && ((Node *)visited)->source != NULL))
    {
        PyErr_SetString(PyExc_TypeError, "Node_MultiVisitor only supports visiting of (properly initialized) Node objects");
        return 0;
    }


    /* Obtain source objects of the node */
    source = ((Node *)visited)->source;
    
    if(source == NULL || source == Py_None || PyList_Size(source) == 0)
        return PyTuple_New(0);
    else
    {
        Py_ssize_t sourcesize = PyList_GET_SIZE(source);
        PyObject *res;
        PyObject *resvisit;
        int i;

        /* Walk through nodes, and store results in a tuple */
        res = PyTuple_New(sourcesize);
        for(i = 0; i < sourcesize; i++)
        {   
            /* Reuse or own args tuple */ 
            PyTuple_SET_ITEM(args,0,PyList_GET_ITEM(source,i));
            resvisit = PyObject_CallMethod((PyObject *)self,PyString_AsString(prefix),"O",args);

            if(resvisit == NULL)
            {
                PyTuple_SET_ITEM(args,0,visited);
                Py_DECREF(res);
                return NULL;
            }
            /* Store results */
            PyTuple_SET_ITEM(res,i,resvisit);
        }
        /* Recover old args tuple */
        PyTuple_SET_ITEM(args,0,visited);
        return res;
    }
}

#define NMV_VISIT(functionnr) \
static PyObject * Node_MultiVisitor_visit##functionnr(Node_MultiVisitor *self,PyObject *args) \
{ \
   if(self->prefixes == NULL || !PyTuple_Check(self->prefixes) || PyTuple_GET_SIZE(self->prefixes) < (functionnr)) \
   { \
        PyErr_SetString(PyExc_TypeError, "Visitor not correctly initialized (no prefixes)."); \
        return NULL; \
   } \
   return Node_MultiVisitor_visit(self,PyTuple_GET_ITEM(self->prefixes,functionnr),args); \
}

NMV_VISIT(0);
NMV_VISIT(1);
NMV_VISIT(2);
NMV_VISIT(3);
NMV_VISIT(4);
NMV_VISIT(5);
NMV_VISIT(6);
NMV_VISIT(7);
NMV_VISIT(8);

#define NMV_VISITSOURCES(functionnr) \
static PyObject * Node_MultiVisitor_visitsources##functionnr(Node_MultiVisitor *self,PyObject *args) \
{ \
   if(self->prefixes == NULL || !PyTuple_Check(self->prefixes) || PyTuple_GET_SIZE(self->prefixes) < (functionnr)) \
   { \
        PyErr_SetString(PyExc_TypeError, "Visitor not correctly initialized (no prefixes)."); \
        return NULL; \
   } \
   return Node_MultiVisitor_visitsources(self,PyTuple_GET_ITEM(self->prefixes,functionnr),args); \
}

NMV_VISITSOURCES(0);
NMV_VISITSOURCES(1);
NMV_VISITSOURCES(2);
NMV_VISITSOURCES(3);
NMV_VISITSOURCES(4);
NMV_VISITSOURCES(5);
NMV_VISITSOURCES(6);
NMV_VISITSOURCES(7);
NMV_VISITSOURCES(8);

#define NMV_TABLE_ENTRY(functionnr) \
    {"__visit" #functionnr, (PyCFunction)Node_MultiVisitor_visit##functionnr, METH_VARARGS, \
     "Visit class by calling appropiate method" \
    }

#define NMVS_TABLE_ENTRY(functionnr) \
    {"__visitsource" #functionnr, (PyCFunction)Node_MultiVisitor_visitsources##functionnr, METH_VARARGS, \
     "Visits source objects of a node and returns result in a tuple" \
    }

/* Currently max 8 extra visit methods */
static PyMethodDef Node_MultiVisitor_methods[] = {
    {"visitsource", (PyCFunction)Node_MultiVisitor_visitsources0, METH_VARARGS,
     "Visits source objects of a node and returns result in a tuple"
    },
    {"visit", (PyCFunction)Node_MultiVisitor_visit0, METH_VARARGS,
     "Visit class by calling appropiate method"
    },
    NMV_TABLE_ENTRY(1),
    NMV_TABLE_ENTRY(2),
    NMV_TABLE_ENTRY(3),
    NMV_TABLE_ENTRY(4),
    NMV_TABLE_ENTRY(5),
    NMV_TABLE_ENTRY(6),
    NMV_TABLE_ENTRY(7),
    NMV_TABLE_ENTRY(8),
    NMVS_TABLE_ENTRY(1),
    NMVS_TABLE_ENTRY(2),
    NMVS_TABLE_ENTRY(3),
    NMVS_TABLE_ENTRY(4),
    NMVS_TABLE_ENTRY(5),
    NMVS_TABLE_ENTRY(6),
    NMVS_TABLE_ENTRY(7),
    NMVS_TABLE_ENTRY(8),
    {NULL} 
};

static PyObject *
Node_MultiVisitor_getParams(Node_MultiVisitor *self, void *closure)
{
    if(self->params == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Params not available (class not correctly initizalized?)."); 
        return NULL;
    }  
    else
    {
        Py_INCREF(self->params);
        return self->params;    
    }
}

static int
Node_MultiVisitor_setParams(Node_MultiVisitor *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(!PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Params should be an list object");
        return -1;
    }
    tmp = self->params;
    Py_INCREF(value);
    self->params = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyGetSetDef Node_MultiVisitor_getseters[] = {
    {"params", 
     (getter)Node_MultiVisitor_getParams, (setter)Node_MultiVisitor_setParams,
     "Get/set query graph parameters",
     NULL},
    {NULL}  /* Sentinel */
};


static PyTypeObject Node_MultiVisitorType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "query_graph.Node_MultiVisitor",             /*tp_name*/
    sizeof(Node_MultiVisitor),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Node_MultiVisitor_dealloc, /*tp_dealloc*/
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
    "Node_MultiVisitor object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    Node_MultiVisitor_methods,         /* tp_methods */
    0,                         /* tp_members */
    Node_MultiVisitor_getseters,    /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Node_MultiVisitor_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};/*}}}*/


static PyTypeObject NodeCache_MultiVisitorType;/*{{{*/

typedef struct {
    Node_MultiVisitor nv;
    PyObject * cache;
} NodeCache_MultiVisitor;

static void
NodeCache_MultiVisitor_dealloc(NodeCache_MultiVisitor* self)
{
    Py_XDECREF(((Node_MultiVisitor *)self)->vcache);
    Py_XDECREF(((Node_MultiVisitor *)self)->params);
    Py_XDECREF(((Node_MultiVisitor *)self)->prefixes);
    Py_XDECREF(self->cache);
    ((Node_MultiVisitor *)self)->ob_type->tp_free((PyObject*)self);
}

static int
NodeCache_MultiVisitor_init(NodeCache_MultiVisitor *self, PyObject *args, PyObject *kwds)
{
    PyObject *cache = NULL;

    if(self->cache == NULL)
    {
        if(kwds != NULL)
            cache = PyDict_GetItemString(kwds,"cache");
        if(cache != NULL)
            cache = PyDict_Copy(cache);
        else
            cache = PyDict_New();
        if(cache == NULL)
            return -1;
        self->cache = cache;
    }
    if(kwds != NULL)
    {
        PyDict_DelItemString(kwds,"cache");
        PyErr_Clear();
    }
    if(Node_MultiVisitor_init((Node_MultiVisitor *)self,args,kwds) < 0)
        return -1;
    return 0;
}

static PyObject * NodeCache_MultiVisitor_visit(NodeCache_MultiVisitor *self,PyObject * prefix,PyObject *args)
{
    Py_ssize_t arglength;
    PyObject * visited;
    PyObject * res, *tmp, *key;

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
   
    if(visited->ob_type != &Node_Type)
        return Node_MultiVisitor_visit((Node_MultiVisitor *)self,prefix,args);
    
    
    if(self->cache == NULL || !PyDict_Check(self->cache))
    {
        PyErr_SetString(PyExc_TypeError, "Class not initialized");
        return 0;
    }   
    key = Py_BuildValue("OO",prefix,(PyObject *) visited);
    if(key == NULL)
        return NULL;

    res = PyDict_GetItem(self->cache,key);
    if(res == NULL)
    {
        res = Node_MultiVisitor_visit((Node_MultiVisitor *)self,prefix,args);
        if(res == NULL)
        {
            Py_DECREF(key);
            return NULL;
        }
        
        if(!PyList_Check(((Node *)visited)->target) ||PyList_GET_SIZE(((Node *)visited)->target)< 2)
        {
            Py_DECREF(key);
            return res;
        }
        if(PyDict_SetItem(self->cache,key,res) < 0)
        {
            Py_DECREF(res);
            Py_DECREF(key);
            return NULL;
        }
    }
    else
    {
        Py_INCREF(res);
    }
    Py_DECREF(key);
   
    //QueryResult objects are copied as the version in the cache may not be changed
    if(res->ob_type == &QueryResult_Type)
    {
        tmp = res;
        res = QueryResult_copy((QueryResult *)res);
        Py_DECREF(tmp);
    }
    return res;
}
/*
static PyObject * NodeCache_MultiVisitor_visitsources(NodeCache_MultiVisitor *self,PyObject *prefix, PyObject *args)
{
    PyObject *visited;
    PyObject *source;
    Py_ssize_t arglength;

    if(!PyTuple_Check(args))
    {
        PyErr_SetString(PyExc_TypeError, "visitsources method should be called with the object to visit.");
        return 0;
    }

    arglength = PyTuple_GET_SIZE(args);
    if(arglength == 0)
    {
        PyErr_SetString(PyExc_TypeError, "visitsources method should be called with the object to visit.");
        return 0;
    }
    
    visited = PyTuple_GET_ITEM(args,0);
    if(visited == NULL)
        return 0;
    if(!(visited->ob_type == &Node_Type && ((Node *)visited)->source != NULL))
    {
        PyErr_SetString(PyExc_TypeError, "Node_MultiVisitor only supports visiting of (properly initialized) Node objects");
        return 0;
    }

    source = ((Node *)visited)->source;
    
    if(source == NULL || source == Py_None || PyList_Size(source) == 0)
        return PyTuple_New(0);
    else
    {
        Py_ssize_t sourcesize = PyList_GET_SIZE(source);
        PyObject *res;
        PyObject *resvisit;
        int i;

        res = PyTuple_New(sourcesize);
        for(i = 0; i < sourcesize; i++)
        {
            PyTuple_SET_ITEM(args,0,PyList_GET_ITEM(source,i));
            resvisit = NodeCache_MultiVisitor_visit(self,prefix,args);
            if(resvisit == NULL)
            {
                PyTuple_SET_ITEM(args,0,visited);
                Py_DECREF(res);
                return NULL;
            }
            PyTuple_SET_ITEM(res,i,resvisit);
        }

        PyTuple_SET_ITEM(args,0,visited);
        return res;
    }
}
*/

#define NCMV_VISIT(functionnr) \
static PyObject * NodeCache_MultiVisitor_visit##functionnr(NodeCache_MultiVisitor *self,PyObject *args) \
{ \
   if(((Node_MultiVisitor *)self)->prefixes == NULL || !PyTuple_Check(((Node_MultiVisitor *)self)->prefixes) || PyTuple_GET_SIZE(((Node_MultiVisitor *)self)->prefixes) < (functionnr)) \
   { \
        PyErr_SetString(PyExc_TypeError, "Visitor not correctly initialized (no prefixes)."); \
        return NULL; \
   } \
   return NodeCache_MultiVisitor_visit(self,PyTuple_GET_ITEM(((Node_MultiVisitor *)self)->prefixes,functionnr),args); \
}

NCMV_VISIT(0);
NCMV_VISIT(1);
NCMV_VISIT(2);
NCMV_VISIT(3);
NCMV_VISIT(4);
NCMV_VISIT(5);
NCMV_VISIT(6);
NCMV_VISIT(7);
NCMV_VISIT(8);

/*
#define NCMV_VISITSOURCES(functionnr) \
static PyObject * NodeCache_MultiVisitor_visitsources##functionnr(NodeCache_MultiVisitor *self,PyObject *args) \
{ \
   if(((Node_MultiVisitor *) self)->prefixes == NULL || !PyTuple_Check(((Node_MultiVisitor *)self)->prefixes) || PyTuple_GET_SIZE(((Node_MultiVisitor *)self)->prefixes) < (functionnr)) \
   { \
        PyErr_SetString(PyExc_TypeError, "Visitor not correctly initialized (no prefixes)."); \
        return NULL; \
   } \
   return NodeCache_MultiVisitor_visitsources(self,PyTuple_GET_ITEM(((Node_MultiVisitor *)self)->prefixes,functionnr),args); \
}

NCMV_VISITSOURCES(0);
NCMV_VISITSOURCES(1);
NCMV_VISITSOURCES(2);
NCMV_VISITSOURCES(3);
NCMV_VISITSOURCES(4);
NCMV_VISITSOURCES(5);
NCMV_VISITSOURCES(6);
NCMV_VISITSOURCES(7);
NCMV_VISITSOURCES(8);
*/

#define NCMV_TABLE_ENTRY(functionnr) \
    {"__visit" #functionnr, (PyCFunction)NodeCache_MultiVisitor_visit##functionnr, METH_VARARGS, \
     "Visit class by calling appropiate method" \
    }

#define NCMVS_TABLE_ENTRY(functionnr) \
    {"__visitsource" #functionnr, (PyCFunction)NodeCache_MultiVisitor_visitsources##functionnr, METH_VARARGS, \
     "Visits source objects of a node and returns result in a tuple" \
    }


static PyMethodDef NodeCache_MultiVisitor_methods[] = {
    {"visit", (PyCFunction)NodeCache_MultiVisitor_visit0, METH_VARARGS,
     "Visit class by calling appropiate method"
    },
//    {"visitsource", (PyCFunction)NodeCache_MultiVisitor_visitsources0, METH_VARARGS,
//     "Visit source of a node"
//    },
    NCMV_TABLE_ENTRY(1),
    NCMV_TABLE_ENTRY(2),
    NCMV_TABLE_ENTRY(3),
    NCMV_TABLE_ENTRY(4),
    NCMV_TABLE_ENTRY(5),
    NCMV_TABLE_ENTRY(6),
    NCMV_TABLE_ENTRY(7),
    NCMV_TABLE_ENTRY(8),
//    NCMVS_TABLE_ENTRY(1),
//    NCMVS_TABLE_ENTRY(2),
//    NCMVS_TABLE_ENTRY(3),
//    NCMVS_TABLE_ENTRY(4),
//    NCMVS_TABLE_ENTRY(5),
//    NCMVS_TABLE_ENTRY(6),
//    NCMVS_TABLE_ENTRY(7),
//    NCMVS_TABLE_ENTRY(8),
    {NULL} 
};

static PyObject *
NodeCache_MultiVisitor_getCache(NodeCache_MultiVisitor *self, void *closure)
{
    if(self->cache == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Visitor not correctly initialized (no cache)."); \
        return NULL;
    }  
    else
    {
        Py_INCREF(self->cache);
        return self->cache;
    }
}

static PyGetSetDef NodeCache_MultiVisitor_getseters[] = {
    {"cache", 
     (getter)NodeCache_MultiVisitor_getCache, NULL,
     "Get node to result cache",
     NULL},
    {NULL}  /* Sentinel */
};




static PyTypeObject NodeCache_MultiVisitorType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "query_graph.NodeCache_MultiVisitor",             /*tp_name*/
    sizeof(NodeCache_MultiVisitor),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)NodeCache_MultiVisitor_dealloc, /*tp_dealloc*/
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
    "NodeCache_MultiVisitor object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    NodeCache_MultiVisitor_methods,         /* tp_methods */
    0,                         /* tp_members */
    NodeCache_MultiVisitor_getseters,  /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)NodeCache_MultiVisitor_init,      /* tp_init */
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
initquery_graph(void) 
{
    PyObject* m;
    ModificationSemaphore_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ModificationSemaphore_Type) < 0)
        return;
    
    QueryResult_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&QueryResult_Type) < 0)
        return;

    BaseContainer_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&BaseContainer_Type) < 0)
        return;

    Node_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&Node_Type) < 0)
        return;
    
    MultiVisitorType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&MultiVisitorType) < 0)
        return;
    
    QGCreator_Type.tp_base = &MultiVisitorType;
    if (PyType_Ready(&QGCreator_Type) < 0)
        return;

    Node_MultiVisitorType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&Node_MultiVisitorType) < 0)
        return;
    
    NodeCache_MultiVisitorType.tp_base = &Node_MultiVisitorType;
    if (PyType_Ready(&NodeCache_MultiVisitorType) < 0)
        return;

    m = Py_InitModule3("query_graph", module_methods,
                       "Code to handle query graphs");

    import_array();
    if (m == NULL)
      return;

    Py_INCREF(&ModificationSemaphore_Type);
    Py_INCREF(&QueryResult_Type);
    Py_INCREF(&BaseContainer_Type);
    Py_INCREF(&Node_Type);
    Py_INCREF(&MultiVisitorType);
    Py_INCREF(&Node_MultiVisitorType);
    Py_INCREF(&NodeCache_MultiVisitorType);
    PyModule_AddObject(m, "ModificationSemaphore", (PyObject *)&ModificationSemaphore_Type);
    PyModule_AddObject(m, "QueryResult", (PyObject *)&QueryResult_Type);
    PyModule_AddObject(m, "BaseContainer", (PyObject *)&BaseContainer_Type);
    PyModule_AddObject(m, "Node", (PyObject *)&Node_Type);
    PyModule_AddObject(m, "MultiVisitor", (PyObject *)&MultiVisitorType);
    PyModule_AddObject(m, "QGCreator", (PyObject *)&QGCreator_Type);
    PyModule_AddObject(m, "Node_MultiVisitor", (PyObject *)&Node_MultiVisitorType);
    PyModule_AddObject(m, "NodeCache_MultiVisitor", (PyObject *)&NodeCache_MultiVisitorType);
}
