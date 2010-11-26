#include "Python.h"
#include "structmember.h"
#include "closure.h"
#include "multi_visitor.h"
#include "base_container.h"


/* Node objects for the query graph. Contain references
 * to their source and target objects, as well as
 * fields, properties, and active fields. */

/*{{{*/static PyTypeObject Node_Type;

typedef struct {
    PyObject_HEAD
    PyObject * obj;
    PyObject * source;
    PyObject * target;

    PyObject * fields;
    PyObject * actidx;
    PyObject * segments;
    PyObject * invar_dict;
    PyObject * dims;
    PyObject * actidx_dims;

    PyObject * in_fields;
    PyObject * out_calc;
    PyObject * out_fields;
    PyObject * exec_params;
    PyObject * req_fields;
} Node;

static void
Node_dealloc(Node* self)
{
    Py_XDECREF(self->obj);
    Py_XDECREF(self->source);
    Py_XDECREF(self->target);
    
    Py_XDECREF(self->fields);
    Py_XDECREF(self->actidx);
    Py_XDECREF(self->segments);
    Py_XDECREF(self->invar_dict);
    Py_XDECREF(self->dims);
    Py_XDECREF(self->actidx_dims);

    Py_XDECREF(self->in_fields);
    Py_XDECREF(self->out_calc);
    Py_XDECREF(self->out_fields);
    Py_XDECREF(self->exec_params);
    Py_XDECREF(self->req_fields);
    self->ob_type->tp_free((PyObject*)self);
}


static int
Node_init(Node *self, PyObject *args, PyObject *kwds)
{
   PyObject *obj,*source = NULL,*target = NULL, *segments;
   Invar * invar;
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


   if(PyObject_IsInstance(obj,(PyObject *)BaseContainer_Type))
   {
        Py_INCREF(obj->ob_type);
        self->obj = (PyObject *)obj->ob_type;
        invar = ((BaseContainer *)obj)->invar;

        if(invar == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Invar should be set");
            return -1;
        }

        
        segments = invar->segments;
        if(segments == NULL)
        {
            segments = PyObject_GetAttrString(self->obj,"_csegments");
            

            if(segments == NULL)
            {
                return -1;
            }
        }
        Py_INCREF(segments);

        if(PyTuple_Check(segments)) //tuple to list (prevents original node to be changed during optimization)
        {
            int i;
            PyObject *nsegments,*item;
            long size = PyTuple_GET_SIZE(segments);;
            nsegments = PyList_New(size);
            for(i = 0; i < size; i++ )
            {
                item = PyTuple_GET_ITEM(segments,i);
                Py_INCREF(item);
                PyList_SET_ITEM(nsegments,i,item);
            }
            Py_DECREF(segments);
            segments = nsegments;
        }
        self->segments = segments;


        Py_XINCREF(invar->fields);
        self->fields = invar->fields;

        Py_XINCREF(invar->actidx);
        self->actidx = invar->actidx;
        
        Py_XINCREF(invar->dims);
        self->dims = invar->dims;
        Py_XINCREF(invar->actidx_dims);
        self->actidx_dims = invar->actidx_dims;

        Py_XINCREF(invar->local);
        self->invar_dict = invar->local;

   }
   else
   {
        Py_INCREF(obj);
        self->obj = obj;
   }

   return 0;
}


static PyObject * Node_getObj(Node *self, void *closure)/*{{{*/
{
    Py_INCREF(self->obj);
    return self->obj;
}
static int Node_setObj(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->obj;
    Py_XINCREF(value);
    self->obj = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getSource(Node *self, void *closure)/*{{{*/
{
    Py_INCREF(self->source);
    return self->source;
}
static int Node_setSource(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL || !PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Source should be a list object");
        return -1;
    }
    tmp = self->source;
    Py_INCREF(value);
    self->source = value;
    Py_DECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getTarget(Node *self, void *closure)/*{{{*/
{
    Py_INCREF(self->target);
    return self->target;
}
static int Node_setTarget(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(value == NULL || !PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Target should be a list object");
        return -1;
    }
    tmp = self->target;
    Py_INCREF(value);
    self->target = value;
    Py_DECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getFields(Node *self, void *closure)/*{{{*/
{
    if(!self->fields)
    {
        Py_INCREF(Py_None);
        self->fields = Py_None;
    }
    Py_INCREF(self->fields);
    return self->fields;
}
static int Node_setFields(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->fields;
    Py_XINCREF(value);
    self->fields = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getActIdx(Node *self, void *closure)/*{{{*/
{
    if(!self->actidx)
    {
        Py_INCREF(Py_None);
        self->actidx = Py_None;
    }
    Py_INCREF(self->actidx);
    return self->actidx;
}
static int Node_setActIdx(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->actidx;
    Py_XINCREF(value);
    self->actidx = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getDims(Node *self, void *closure)/*{{{*/
{
    if(!self->dims)
    {
        Py_INCREF(Py_None);
        self->dims = Py_None;
    }
    Py_INCREF(self->dims);
    return self->dims;
}
static int Node_setDims(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->dims;
    Py_XINCREF(value);
    self->dims = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/
static PyObject * Node_getActIdxDims(Node *self, void *closure)/*{{{*/
{
    if(!self->actidx_dims)
    {
        Py_INCREF(Py_None);
        self->actidx_dims = Py_None;
    }
    Py_INCREF(self->actidx_dims);
    return self->actidx_dims;
}
static int Node_setActIdxDims(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->actidx_dims;
    Py_XINCREF(value);
    self->actidx_dims = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/


static PyObject * Node_getSegments(Node *self, void *closure)/*{{{*/
{
    if(!self->segments)
    {
        Py_INCREF(Py_None);
        self->segments = Py_None;
    }
    Py_INCREF(self->segments);
    return self->segments;
}
static int Node_setSegments(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->segments;
    Py_XINCREF(value);
    self->segments = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getInFields(Node *self, void *closure)/*{{{*/
{
    if(!self->in_fields)
    {
        Py_INCREF(Py_None);
        self->in_fields = Py_None;
    }
    Py_INCREF(self->in_fields);
    return self->in_fields;
}
static int Node_setInFields(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->in_fields;
    Py_XINCREF(value);
    self->in_fields = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getOutCalc(Node *self, void *closure)/*{{{*/
{
    if(!self->out_calc)
    {
        Py_INCREF(Py_None);
        self->out_calc = Py_None;
    }
    Py_INCREF(self->out_calc);
    return self->out_calc;
}
static int Node_setOutCalc(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->out_calc;
    Py_XINCREF(value);
    self->out_calc = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getOutFields(Node *self, void *closure)/*{{{*/
{
    if(!self->out_fields)
    {
        Py_INCREF(Py_None);
        self->out_fields = Py_None;
    }
    Py_INCREF(self->out_fields);
    return self->out_fields;
}
static int Node_setOutFields(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->out_fields;
    Py_XINCREF(value);
    self->out_fields = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getExecParams(Node *self, void *closure)/*{{{*/
{
    if(!self->exec_params)
    {
        Py_INCREF(Py_None);
        self->exec_params = Py_None;
    }
    Py_INCREF(self->exec_params);
    return self->exec_params;
}
static int Node_setExecParams(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->exec_params;
    Py_XINCREF(value);
    self->exec_params = value;
    Py_XDECREF(tmp);
    return 0;
}/*}}}*/

static PyObject * Node_getReqFields(Node *self, void *closure)/*{{{*/
{
    if(!self->req_fields)
    {
        Py_INCREF(Py_None);
        self->req_fields = Py_None;
    }
    Py_INCREF(self->req_fields);
    return self->req_fields;
}
static int Node_setReqFields(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->req_fields;
    Py_XINCREF(value);
    self->req_fields = value;
    Py_XDECREF(tmp);
    return 0;
}
/*}}}*/

static PyObject * Node_getInvarDict(Node *self, void *closure)/*{{{*/
{
    if(!self->invar_dict)
    {
        Py_INCREF(Py_None);
        self->invar_dict = Py_None;
    }
    Py_INCREF(self->invar_dict);
    return self->invar_dict;
}
static int Node_setInvarDict(Node *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    tmp = self->invar_dict;
    Py_XINCREF(value);
    self->invar_dict = value;
    Py_XDECREF(tmp);
    return 0;
}
/*}}}*/

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
    {"actidx", 
     (getter)Node_getActIdx, (setter)Node_setActIdx,
     "Active field indexes",
     NULL},
    {"dims", 
     (getter)Node_getDims, (setter)Node_setDims,
     "Dimensions represented by this node",
     NULL},
    {"actidx_dims", 
     (getter)Node_getActIdxDims, (setter)Node_setActIdxDims,
     "Active dimension indexes",
     NULL},
    {"segments", 
     (getter)Node_getSegments, (setter)Node_setSegments,
     "Segments",
     NULL},
    {"in_fields", 
     (getter)Node_getInFields, (setter)Node_setInFields,
     "List of fields required by this node for each source",
     NULL},
    {"out_calc", 
     (getter)Node_getOutCalc, (setter)Node_setOutCalc,
     "Specifies which derivative fields to calculate.",
     NULL},
    {"out_fields", 
     (getter)Node_getOutFields, (setter)Node_setOutFields,
     "Fields output by this node.",
     NULL},
    {"exec_params", 
     (getter)Node_getExecParams, (setter)Node_setExecParams,
     "Extra execution parameters",
     NULL},
    {"req_fields", 
     (getter)Node_getReqFields, (setter)Node_setReqFields,
     "Extra required fields",
     NULL},
    {"invar_dict", 
     (getter)Node_getInvarDict, (setter)Node_setInvarDict,
     "Invariant dictionary",
     NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject Node_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "qgraph.Node",             /*tp_name*/
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

/*{{{*/
typedef struct {
    PyDWBaseObject base;
    PyObject * params;
    PyObject * objs;
    PyObject * qg_node;
    PyObject * cache;
    
    PyObject * modifiables;
    PyObject * cur_sems;

    long cur_param_pos;
    long max_param_length;
} QGCreator;

static void
QGCreator_dealloc(QGCreator* self)
{
    Py_XDECREF(self->params);
    Py_XDECREF(self->objs);
    Py_XDECREF(self->qg_node);
    Py_XDECREF(self->cache);
    
    Py_XDECREF(self->modifiables);
    Py_XDECREF(self->cur_sems);
    PyDWBaseObject_Type->tp_dealloc((PyObject *)self);

}

static PyObject *qgc_visit_closure;
static PyObject * QGCreator_visit(QGCreator * self, PyObject * args)
{
    PyObject *visited, *source,*target;
    PyObject *res, *item;
    Py_ssize_t ssize;
    int i;
    if(PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of arguments given to visit function.");
        return NULL;
    }
    visited = PyTuple_GET_ITEM(args,0);
   

    if(PyObject_IsInstance(visited,(PyObject *) BaseContainer_Type))
    {
        if(self->cache == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cache not available.");
            return NULL;
        }
        res = PyDict_GetItem(self->cache,visited);
        if(res == NULL)
        {
            res = Visitor_visit((PyObject *)self,args,qgc_visit_closure);
            if(res == NULL)
                return NULL;
            
            //check if target is not initialized
            if(res->ob_type != &Node_Type)
            {
                PyErr_SetString(PyExc_TypeError, "Visit function should return Node object for Containers.");
                return NULL;
            }

            source = ((Node *) res)->source;
            if(source == NULL)
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
                    if(target == NULL)
                    {
                        PyErr_SetString(PyExc_TypeError, "Node object has not been initialized correctly.");
                        return NULL;
                    }
                    PyList_Append(target,res);
                }
            }
            
            /* Handle modifiable containers */
            if(((BaseContainer *)visited)->invar == NULL)
            {
                PyErr_SetString(PyExc_TypeError, "Invar has not been set.");
                return NULL;
 
            }
            if(((Invar *)((BaseContainer *)visited)->invar)->modify_sems != NULL)
            {
                PyList_Append(self->modifiables,visited);
                if(self->cur_sems == NULL)
                {
                    self->cur_sems = PySet_New(((Invar*)((BaseContainer *)visited)->invar)->modify_sems);
                }
                else
                {
                    self->cur_sems = PyNumber_InPlaceAnd(self->cur_sems,((Invar*)((BaseContainer *)visited)->invar)->modify_sems);
                    if(self->cur_sems == NULL)
                        return NULL;
                    Py_DECREF(self->cur_sems);
                }
            }

            /* store visited object types */
            if(self->objs == NULL)
            {
                PyErr_SetString(PyExc_TypeError, "QGCreator has not been properly initialized (objs == NULL).");
                return NULL;
            }
            if(PySet_Add(self->objs,(PyObject *)visited->ob_type) < 0)
                return NULL;

            if(PyDict_SetItem(self->cache,visited,res)< 0)
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


static PyObject *qgc_param_closure;
static PyObject * QGCreator_param(QGCreator * self, PyObject * args)
{
    PyObject *visited;
    PyObject *res;
    
    if(PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of arguments given to visit function.");
        return NULL;
    }
    visited = PyTuple_GET_ITEM(args,0);

    if(PyObject_IsInstance(visited,(PyObject *) BaseContainer_Type))
    {
        if(self->cache == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "Cache not available.");
            return NULL;
        }
        res = PyDict_GetItem(self->cache,visited);
        if(res == NULL)
        {
            res = Visitor_visit((PyObject *) self,args,qgc_param_closure);
            if(res == NULL)
                return NULL;
            if(PyDict_SetItem(self->cache,visited,res)< 0)
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

static int QGCreator_init(QGCreator *self, PyObject *args, PyObject *kwds)
{
    PyObject *tmp, *root, *nargs;
    
    static char *kwlist[] = {"root",NULL};
    if(!PyArg_ParseTupleAndKeywords(args,kwds,"O",kwlist,
            &root))
      return -1;

    if(!PyObject_IsInstance(root,(PyObject *) BaseContainer_Type))
    {
        PyErr_SetString(PyExc_TypeError, "First parameter should be a BaseContainer object");
        return -1;
    }
    /* reset cache */
    tmp = self->cache;
    self->cache = PyDict_New();
    Py_XDECREF(tmp);
   
    /* reset parameters */
    tmp = self->params;
    self->params = PyList_New(0);
    Py_XDECREF(tmp);

    /* reset obj type list */
    tmp = self->objs;
    self->objs = PySet_New(NULL);
    Py_XDECREF(tmp);
    if(self->cache == NULL || self->params == NULL || self->objs == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QGCreator object not properly initialized.");
        return -1;
    }

    /* reset modifiable state */
    tmp = self->modifiables;
    self->modifiables = PyList_New(0);
    Py_XDECREF(tmp);
    Py_XDECREF(self->cur_sems);
    self->cur_sems = NULL;
    
    Py_XDECREF(self->qg_node);

    nargs = PyTuple_Pack(2,root,root);
    if(nargs == NULL)
        return -1;
    self->qg_node = QGCreator_visit(self,nargs);
    
    /* drop cache content (so that query object can be deleted) */
    PyDict_Clear(self->cache);
    Py_DECREF(nargs);
    
    if(self->qg_node == NULL)
        return -1;

    return 0;
}


static PyObject * QGCreator_update_graph(QGCreator *self,PyObject *root)
{
    PyObject *tmp,*args, *res;
    if(!PyObject_IsInstance(root,(PyObject *) BaseContainer_Type))
    {
        PyErr_SetString(PyExc_TypeError, "First parameter should be a BaseContainer object");
        return NULL;
    }
    
    /* reset parameters */
    tmp = self->params;
    if(tmp != NULL && PyList_Check(tmp))
        self->max_param_length = PyList_GET_SIZE(tmp);
    else
        self->max_param_length = 0;

    self->params = PyList_New(self->max_param_length);
    self->cur_param_pos = 0;
    Py_XDECREF(tmp);
    if(self->params == NULL || self->cache == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QGCreator object not properly initialized.");
        return NULL;
    }

    args = PyTuple_Pack(2,root,root);
    if(args == NULL)
        return NULL;
    res = QGCreator_param(self,args);
    
    /* drop cache (so that query object can be deleted) */
    PyDict_Clear(self->cache);
    
    Py_DECREF(args);
    
    if(res == NULL)
        return NULL;
    Py_DECREF(res);

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
    PyObject *visited, *source, *source_fast, * item, *nargs, *res, *root;
    int nsource, i;

    if(PyTuple_GET_SIZE(args) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of arguments given to visit function.");
        return NULL;
    }
    visited = PyTuple_GET_ITEM(args,0);
    root = PyTuple_GET_ITEM(args,1);

    if(!PyObject_IsInstance(visited,(PyObject *) BaseContainer_Type))
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect argument type given to visit function.");
        return NULL;
    }
    source = PyObject_GetAttrString(visited,"_source");
    if(source == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "MultiCon container has no _source parameter.");
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
        nargs = PyTuple_Pack(2,item,root);
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
    {"update", (PyCFunction)QGCreator_update_graph, METH_O,
     "Update query graph param vector for new root"
    },
    {"visit", (PyCFunction)QGCreator_visit, METH_VARARGS,
     "Overloaded visit function for qg construction"
    },
    {"param", (PyCFunction)QGCreator_param, METH_VARARGS,
     "Overloaded param function for qg construction"
    },
    {"paramMultiOpCon", (PyCFunction)QGCreator_paramMultiCon, METH_VARARGS,
     "Overloaded param function for MultiCon classes."
    },
    {NULL} 
};


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
QGCreator_getQG(QGCreator *self, void *closure)
{
    if(self->qg_node == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "QG object has not been set(bug).");
        return NULL;
    }
    
    Py_INCREF(self->qg_node);
    return self->qg_node;
}
static int
QGCreator_setQG(QGCreator *self, PyObject * value, void *closure)
{
    PyObject * tmp;
    tmp = self->qg_node;
    Py_INCREF(value);
    self->qg_node = value;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
QGCreator_getParams(QGCreator *self, void *closure)
{
    if(self->params == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Param object has not been set(bug).");
        return NULL;
    }
    
    Py_INCREF(self->params);
    return self->params;
}

static PyObject *
QGCreator_getObjs(QGCreator *self, void *closure)
{
    if(self->objs == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Object types have not been set(bug).");
        return NULL;
    }
    
    Py_INCREF(self->objs);
    return self->objs;
}

static int
QGCreator_setModifiables(QGCreator *self, PyObject *value, void *closure)
{
    PyObject *tmp;
    if(!PyList_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "Modifiables should be a list object");
        return -1;
    }
    tmp = self->modifiables;
    Py_INCREF(value);
    self->modifiables = value;
    Py_XDECREF(tmp);
    return 0;
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
    {"modifiables", 
     (getter)QGCreator_getModifiables, (setter)QGCreator_setModifiables,
     "Obtain modifiable containers in query graph",
     NULL},
    {"_cur_sems", 
     (getter)QGCreator_getCurSems, NULL,
     "Obtain semaphore selection (only for internal use).",
     NULL},
    {"qg", 
     (getter)QGCreator_getQG, (setter)QGCreator_setQG,
     "Obtain root query graph node.",
     NULL},
    {"flags", 
     (getter)QGCreator_getObjs, NULL,
     "Obtain list of object types part of query",
     NULL},
    {"params", 
     (getter)QGCreator_getParams, NULL,
     "Obtain params for query.",
     NULL},
    {NULL}  /* Sentinel */
};



static PyTypeObject QGCreator_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "qgraph.QGCreator",             /*tp_name*/
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


static PyTypeObject * QueryBase_prepare()
{
    PyTypeObject * base;
    PyObject *name, *prefixes, *bases, *flags, *args;
    PyObject *tmp;
    if(QGCreator_Type.tp_base != 0)
    {
        PyErr_SetString(PyExc_TypeError, "QueryBase_prepare should only be called once.");
        return NULL;
    }

    /* Prepare argumetns */
    name = PyString_FromString("QueryBaseBase");
    prefixes = Py_BuildValue("ss","visit","param");
    bases = PyTuple_Pack(1,PyDWBaseObject_Type);
    flags = PyInt_FromLong(0);
    if(name == NULL || prefixes == NULL || bases == NULL || flags == NULL)
        return NULL;
    args = PyTuple_Pack(4,name,prefixes,bases,flags);
    if(args == NULL)
        return NULL;

    /* Check if function is there, then call it */
    if(VisitorFactory == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Could not find VisitorFactory function.");
        return NULL;
    }
    base = (PyTypeObject *) PyObject_Call(VisitorFactory,args,NULL);
    if(base == NULL)
        return NULL;

    /* Now get references to closure objects from new type */
    if(base->tp_dict == NULL || !PyDict_Check(base->tp_dict))
    {
        PyErr_SetString(PyExc_TypeError, "Base type dictionary is not available");
        return NULL;
    }
    tmp = PyDict_GetItemString(base->tp_dict,"visit");
    if(tmp == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Visit function not found.");
        return NULL;
    }
    qgc_visit_closure = PyObject_GetAttrString(tmp,"__closure__");
    tmp = PyDict_GetItemString(base->tp_dict,"param");
    if(tmp == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Param function not found.");
        return NULL;
    }
    qgc_param_closure = PyObject_GetAttrString(tmp,"__closure__");
    if(qgc_param_closure == NULL || qgc_visit_closure == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "CLosure vars could not be obtained.");
        return NULL;
    }
    
    Py_INCREF(qgc_visit_closure);
    Py_INCREF(qgc_param_closure);
     /*QGCreator is base object */
    QGCreator_Type.tp_new = PyType_GenericNew;
    QGCreator_Type.tp_base = base;
    if (PyType_Ready(&QGCreator_Type) < 0)
        return NULL;
   return &QGCreator_Type;

}

/*}}}*/


/* Visitor class for nodes, adding support for parameters
 * by adding a params parameter */
static PyTypeObject NodeVisitor_Type;/*{{{*/ 

typedef struct {
    PyObject_HEAD
    PyObject * params;
} NodeVisitor;

static void
NodeVisitor_dealloc(NodeVisitor* self)
{
    Py_XDECREF(self->params);
    self->ob_type->tp_free((PyObject*)self);
}


/* Initialize visitor with prefixes */
static int
NodeVisitor_init(NodeVisitor *self, PyObject *args, PyObject *kwds)
{
    PyObject *params = NULL, *tmp;

    static char *kwlist[] = {"params",NULL};
    if(!PyArg_ParseTupleAndKeywords(args,kwds,"|O!",kwlist,
            (PyObject *)&PyList_Type,&params))
      return -1;
    
    tmp = self->params;
    Py_XINCREF(params);
    self->params = params;
    Py_XDECREF(tmp);
    return 0;
}

static PyObject *
NodeVisitor_getParams(NodeVisitor *self, void *closure)
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
NodeVisitor_setParams(NodeVisitor *self, PyObject *value, void *closure)
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

static PyGetSetDef NodeVisitor_getseters[] = {
    {"params", 
     (getter)NodeVisitor_getParams, (setter)NodeVisitor_setParams,
     "Get/set query graph parameters",
     NULL},
    {NULL}  /* Sentinel */
};


static PyTypeObject NodeVisitor_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "qgraph.NodeVisitor",             /*tp_name*/
    sizeof(NodeVisitor),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)NodeVisitor_dealloc, /*tp_dealloc*/
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
    "NodeVisitor object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,                     /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,         /* tp_methods */
    0,                         /* tp_members */
    NodeVisitor_getseters,    /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)NodeVisitor_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};/*}}}*/


/* Function creating visitor class. Uses VisitorFactory in multi_visitor to
 * do the actual work. The code here adds the NodeVisitor class to the bases
 * if it is not yet there */
static PyObject * NodeVisitorFactory(PyObject *self, PyObject *args, PyObject *kwds, PyObject *closure)/*{{{*/
{
    PyObject *bases = NULL, *nargs, *tmp, *item, *res;
    int i, found,bsize;
    /* Get bases if it is there */
    if(args != NULL && PyTuple_GET_SIZE(args) >= 3)
        bases = PyTuple_GET_ITEM(args,2);
    else
        if(kwds != NULL)
            bases = PyDict_GetItemString(kwds,"bases");

    /* Add NodeVisitor_Type if necessary */
    if(bases == NULL)
        bases = PyTuple_Pack(1,&NodeVisitor_Type);
    else
    {
        if(!PyTuple_Check(bases))
        {
            PyErr_SetString(PyExc_TypeError, "Bases parameter should be a tuple.");
            return NULL;
        }
        //Determine if NodeVisitor_Type is already in tuple
        found = 0;
        bsize = PyTuple_GET_SIZE(bases);
        for(i = 0; i < bsize; i++)
        {
            item = PyTuple_GET_ITEM(bases,i);
            if(PyType_Check(item) && PyObject_IsSubclass(item,(PyObject *) &NodeVisitor_Type))
            {
                found = 1;
                break;
            }
        }
        if(!found)
        {
           tmp = PyTuple_Pack(1,&NodeVisitor_Type);
           if(tmp == NULL)
                return NULL;
           //we can just overwrite bases, it is not our own ref
           bases = PySequence_Concat(tmp,bases);
           Py_DECREF(tmp);
        }
        else
            Py_INCREF(bases);
    }
    if(bases == NULL) // error checking
        return NULL;

    /* Now place bases back */
    if(args != NULL && PyTuple_GET_SIZE(args) >= 3)
    {
        //create copy of args tuple with bases replaced
        bsize = PyTuple_GET_SIZE(args);
        nargs = PyTuple_New(bsize);
        for(i = 0; i < bsize; i++)
        {
            if(i == 2)
                //bases ref is our own
                PyTuple_SetItem(nargs,2,bases); 
            else
            {
                tmp = PyTuple_GET_ITEM(args,i);
                Py_INCREF(tmp);
                PyTuple_SetItem(nargs,i,tmp);
            }
        }
        res = Visitor_factory(self,nargs,kwds,closure);
        Py_DECREF(nargs);
    }
    else
    {
        if(kwds == NULL)
        {
            kwds = PyDict_New();
            if(kwds == NULL)
                goto QVF_error;
        }

        
        if(PyDict_SetItemString(kwds,"bases",bases) < 0)
            goto QVF_error;
        Py_DECREF(bases);  //we had a ref to bases ourself
        res = Visitor_factory(self,args,kwds,closure);
    }
    return res;

QVF_error:
    Py_DECREF(bases);
    return NULL;
}/*}}}*/

/* Function overlaoding the visitkey functionality of the standard visitor. 
 * Has special functionality for the param array. If it encounters a
 * QueryResult object it makes a copy before going on. */
static PyObject * NodeVisitor_visitkey(NodeVisitor * self, PyObject ** pvisited)/*{{{*/
{

    PyObject * visitid;
    /* If the value is an int, it is an index into the param array. */
    if(PyInt_Check(*pvisited))
    {
        /* Check if there is actually a good, long enough, param list available */
        long pidx = PyInt_AsLong(*pvisited);
        if(self->params == NULL || (PyList_GET_SIZE(self->params)) <= pidx)
        {
            PyErr_SetString(PyExc_TypeError, "Paramlist array has not been (correctly) initialized.");
            return 0;
        }

        /* update object to visit*/
        *pvisited = PyList_GET_ITEM(self->params,pidx);
        if(*pvisited == NULL)
            return NULL;
    }    
    
    //QueryResult objects are copied as there version in the query graph tree
    //may not be changed
    if((*pvisited)->ob_type == &Node_Type)
    {
        Py_INCREF(*pvisited);
        visitid = ((Node *)(*pvisited))->obj;
    }
    else
    {
        
        if((*pvisited)->ob_type == QueryResult_Type)
            *pvisited = QueryResult_copy((QueryResult *)(*pvisited));
        else
            Py_INCREF(*pvisited);
        visitid = (PyObject *)(*pvisited)->ob_type;
    }   
    Py_INCREF(visitid);

    return visitid;
}/*}}}*/

/* Function overloading the findmethod functionality in the standard visitor. 
 * Adds code to handle string objects by looking for methods with name prefix<string>. 
 */
static PyObject * NodeVisitor_findmethod(PyObject *self, PyObject *prefix, PyObject *visitkey)/*{{{*/
{
    PyObject *curname, *visit_method;

    if(PyString_Check(visitkey))
    {
        /* Create method name */
        curname = PyString_FromFormat("%s%s",PyString_AS_STRING(prefix),PyString_AS_STRING(visitkey));
        if(curname == NULL)
        {
            return NULL;
        }
        visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
        Py_DECREF(curname);
        if(visit_method == NULL)
            PyErr_Clear();
    }
    else
        visit_method = Visitor_findmethod(self,prefix,visitkey);

    return visit_method;
}/*}}}*/

/* Function overloading the notfound functionality of the standard visitor. It first looks
 * if there are <prefix>node or <prefix>param functions. Otherwise, it uses default
 * functionality of multi_visitor */
static PyObject * NodeVisitor_notfound(PyObject *self, PyObject * prefix, PyObject *visited, long flags)/*{{{*/
{
    PyObject *curname, *visit_method = NULL;
    if(visited->ob_type == &Node_Type)
        curname = PyString_FromFormat("%s%s",PyString_AS_STRING(prefix),"node");
    else
        curname = PyString_FromFormat("%s%s",PyString_AS_STRING(prefix),"param");

    if(curname == NULL)
        return NULL;
    visit_method = PyObject_GetAttr((PyObject *) self->ob_type,curname);
    Py_DECREF(curname);
    if(visit_method == NULL)
    {
        PyErr_Clear();
        if(visited->ob_type == &Node_Type)
            return Visitor_notfound(self,prefix,visited,flags);
        else
        {
           Py_INCREF(Py_None);
           visit_method = Py_None;
           return visit_method;
        }
    }
    else
        return visit_method;
}/*}}}*/

/* Function to bisit the sources of a node */
static PyObject * NodeVisitor_visitsources(NodeVisitor *self, PyObject *args, PyObject *closure)/*{{{*/
{
    PyObject *visited;
    PyObject *source;
    PyObject *prefix;
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
    prefix = PyTuple_GET_ITEM(closure,0); 
    if(!PyList_Check(source) || PyList_Size(source) == 0)
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
            //resvisit = Visitor_visit((PyObject *)self,args,closure);

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
}/*}}}*/

//Method definitions used by the NodeVisitor_visitify function
static PyMethodDef object_methods[] = {
    //visit function comes from multi_visitor, so has to be filled in 
    //in the init function
    {"visit",NULL, METH_VARARGS, "Visitor method"},
    {"visitsources",(PyCFunction) NodeVisitor_visitsources, METH_VARARGS, "Visits source objects of a node"},
    {NULL}  /* Sentinel */
};

/* Function overloading the visitify function of multi_visitor, adding
 * the addition of the <prefix>sources function */
static int NodeVisitor_visitify(PyObject * prefix, /*{{{*/
    PyTypeObject * type, PyObject * flags, PyObject *closure)
{

    PyObject *dict, *descr, *nclosure, *vkfuncobj, *fmfuncobj, *nffuncobj;

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
    if(descr == NULL)
        return -1;

    if(PyDict_SetItem(dict,prefix,descr)<0)
    {
        Py_DECREF(descr);
        return -1;
    }
    Py_DECREF(descr);
    
    descr = PyDescr_NewClosureMethod(type,&object_methods[1],nclosure);
    if(descr == NULL)
        return -1;
   
    if(PyDict_SetItem(dict,PyString_FromFormat("%s%s",PyString_AS_STRING(prefix),"sources"),descr)<0)
    {
        Py_DECREF(descr);
        return -1;
    }
    Py_DECREF(descr);
    Py_DECREF(nclosure);

    return 0;
}/*}}}*/


//Method definition to construction the NodeVisitorFactory function
static PyMethodDef closure_methods[] = {
    {"NodeVisitorFactory",(PyCFunction) NodeVisitorFactory, METH_KEYWORDS, "Creates node visitor classes"},
    {NULL}  /* Sentinel */
};
static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};



#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initqgraph(void) 
{
    PyObject* m;
    PyObject *closure;
    PyObject *visitfactoryfunc;
    PyTypeObject *query_base;
    m = Py_InitModule3("qgraph", module_methods,
                       "Query Graph related code");
    if (m == NULL)
      return;
    import_closure();
    import_multi_visitor();
    import_base_container();

    object_methods[0].ml_meth = (PyCFunction)Visitor_visit;

    /* Add closure funciton object for visitfactory function */
    //replaceable functions used by the visit machinery
    closure = create_closure((VisitKeyFunction) NodeVisitor_visitkey, NodeVisitor_findmethod,
                    NodeVisitor_visitify, NodeVisitor_notfound);
    if(closure == NULL)
        return;
    visitfactoryfunc = PyClosureFunction_New(&closure_methods[0],m,m,closure);
    if(visitfactoryfunc == NULL)
        return;
    Py_DECREF(closure);

    Node_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&Node_Type) < 0)
        return;
    NodeVisitor_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&NodeVisitor_Type) < 0)
        return;
    
    query_base = QueryBase_prepare();
    if(query_base == NULL)
        return;

    //add to module
    Py_INCREF(&Node_Type);
    Py_INCREF(query_base);
    PyModule_AddObject(m,"NodeVisitorFactory",visitfactoryfunc);
    PyModule_AddObject(m,"QueryBase",(PyObject *) query_base);
    PyModule_AddObject(m,"Node", (PyObject *)&Node_Type);
}
