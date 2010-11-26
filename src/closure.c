/* Note: Code based on python source, which is GPL-compatible licensed,
 * but it is not required to relicense this code under GPL as 
 * far as I understand. */


#include "Python.h"
#include "structmember.h"

#define CLOSURE_MODULE
#include "closure.h"

/* WRITE_RESTRICTED was renamed to PY_WRITE_RESTRICTED in Python 2.6.
 * To make this code compatible with Python 2.6, we alias
 * PY_WRITE_RESTRICTED to WRITE_RESTRICTED. */
#ifndef WRITE_RESTRICTED
#define WRITE_RESTRICTED PY_WRITE_RESTRICTED
#endif

/* Free list for method objects to safe malloc/free overhead
 * The m_self element is used to chain the objects.
 */

PyAPI_DATA(PyTypeObject) PyClosureFunction_Type;

#define PyClosureFunction_Check(op) ((op)->ob_type == &PyClosureFunction_Type)
/* Macros for direct access to these values. Type checks are *not*
   done, so use with care. */
#define PyClosureFunction_GET_FUNCTION(func) \
        (((PyClosureFunctionObject *)func) -> m_ml -> ml_meth)
#define PyClosureFunction_GET_SELF(func) \
	(((PyClosureFunctionObject *)func) -> m_self)
#define PyClosureFunction_GET_CLOSURE(func) \
	(((PyClosureFunctionObject *)func) -> m_closure)
#define PyClosureFunction_GET_FLAGS(func) \
	(((PyClosureFunctionObject *)func) -> m_ml -> ml_flags)

typedef PyObject *(*PyClosureFunction)(PyObject *, PyObject *, PyObject *);
typedef PyObject *(*PyClosureFunctionWithKeywords)(PyObject *, PyObject *, PyObject *,PyObject *);
typedef PyObject *(*PyClosureNoArgFunction)(PyObject *, PyObject *);

typedef struct {
    PyObject_HEAD
    PyMethodDef *m_ml; /* Description of the C function to call */
    PyObject    *m_self; /* Passed as 'self' arg to the C func, can be NULL */
    PyObject    *m_closure; /* Passed as closure arg to the C function */
    PyObject    *m_module; /* The __module__ attribute, can be anything */
} PyClosureFunctionObject;

static PyClosureFunctionObject *free_list = NULL;
static int numfree = 0;
#ifndef PyClosureFunction_MAXFREELIST
#define PyClosureFunction_MAXFREELIST 256
#endif

static PyObject *
PyClosureFunction_New(PyMethodDef *ml, PyObject *self, PyObject *module, PyObject *closure)
{
	PyClosureFunctionObject *op;
	op = free_list;
	if (op != NULL) {
		free_list = (PyClosureFunctionObject *)(op->m_self);
		PyObject_INIT(op, &PyClosureFunction_Type);
		numfree--;
	}
	else {
		op = PyObject_GC_New(PyClosureFunctionObject, &PyClosureFunction_Type);
		if (op == NULL)
			return NULL;
	}
	op->m_ml = ml;
	Py_XINCREF(self);
	op->m_self = self;
    Py_XINCREF(closure);
    op->m_closure = closure;
	Py_XINCREF(module);
	op->m_module = module;
	PyObject_GC_Track(op);
    
	return (PyObject *)op;
}

static PyClosureFunction
PyClosureFunction_GetFunction(PyObject *op)
{
	if (!PyClosureFunction_Check(op)) {
		PyErr_BadInternalCall();
		return NULL;
	}
	return (PyClosureFunction) ((PyClosureFunctionObject *)op) -> m_ml -> ml_meth;
}

static PyObject *
PyClosureFunction_GetSelf(PyObject *op)
{
	if (!PyClosureFunction_Check(op)) {
		PyErr_BadInternalCall();
		return NULL;
	}
	return ((PyClosureFunctionObject *)op) -> m_self;
}

static int
PyClosureFunction_GetFlags(PyObject *op)
{
	if (!PyClosureFunction_Check(op)) {
		PyErr_BadInternalCall();
		return -1;
	}
	return ((PyClosureFunctionObject *)op) -> m_ml -> ml_flags;
}

static PyObject *
PyClosureFunction_Call(PyObject *func, PyObject *arg, PyObject *kw)
{
	PyClosureFunctionObject* f = (PyClosureFunctionObject*)func;
	PyClosureFunction meth = (PyClosureFunction) PyClosureFunction_GET_FUNCTION(func);
	PyObject *self = PyClosureFunction_GET_SELF(func);
	PyObject *closure = PyClosureFunction_GET_CLOSURE(func);
	Py_ssize_t size;

	switch (PyClosureFunction_GET_FLAGS(func) & ~(METH_CLASS | METH_STATIC | METH_COEXIST)) {
	case METH_VARARGS:
		if (kw == NULL || PyDict_Size(kw) == 0)
			return (*meth)(self, arg, closure);
		break;
	case METH_VARARGS | METH_KEYWORDS:
	case METH_OLDARGS | METH_KEYWORDS:
		return (*(PyClosureFunctionWithKeywords)meth)(self, arg, kw, closure);
	case METH_NOARGS:
		if (kw == NULL || PyDict_Size(kw) == 0) {
			size = PyTuple_GET_SIZE(arg);
			if (size == 0)
				return (*meth)(self, NULL, closure);
			PyErr_Format(PyExc_TypeError,
			    "%.200s() takes no arguments (%zd given)",
			    f->m_ml->ml_name, size);
			return NULL;
		}
		break;
	case METH_O:
		if (kw == NULL || PyDict_Size(kw) == 0) {
			size = PyTuple_GET_SIZE(arg);
			if (size == 1)
				return (*meth)(self, PyTuple_GET_ITEM(arg, 0),closure);
			PyErr_Format(PyExc_TypeError,
			    "%.200s() takes exactly one argument (%zd given)",
			    f->m_ml->ml_name, size);
			return NULL;
		}
		break;
	case METH_OLDARGS:
		/* the really old style */
		if (kw == NULL || PyDict_Size(kw) == 0) {
			size = PyTuple_GET_SIZE(arg);
			if (size == 1)
				arg = PyTuple_GET_ITEM(arg, 0);
			else if (size == 0)
				arg = NULL;
			return (*meth)(self, arg, closure);
		}
		break;
	default:
		PyErr_BadInternalCall();
		return NULL;
	}
	PyErr_Format(PyExc_TypeError, "%.200s() takes no keyword arguments",
		     f->m_ml->ml_name);
	return NULL;
}

/* Methods (the standard built-in methods, that is) */

static void
meth_dealloc(PyClosureFunctionObject *m)
{
    PyObject_GC_UnTrack(m);
	Py_XDECREF(m->m_self);
	Py_XDECREF(m->m_closure);
	Py_XDECREF(m->m_module);
	if (numfree < PyClosureFunction_MAXFREELIST) {
		m->m_self = (PyObject *)free_list;
		free_list = m;
		numfree++;
	}
	else {
		PyObject_GC_Del(m);
	}
}

static PyObject *
meth_get__doc__(PyClosureFunctionObject *m, void *closure)
{
	const char *doc = m->m_ml->ml_doc;

	if (doc != NULL)
		return PyString_FromString(doc);
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
meth_get__name__(PyClosureFunctionObject *m, void *closure)
{
	return PyString_FromString(m->m_ml->ml_name);
}

static int
meth_traverse(PyClosureFunctionObject *m, visitproc visit, void *arg)
{
	Py_VISIT(m->m_self);
    Py_VISIT(m->m_closure);
	Py_VISIT(m->m_module);
	return 0;
}

static PyObject *
meth_get__self__(PyClosureFunctionObject *m, void *closure)
{
	PyObject *self;
	if (PyEval_GetRestricted()) {
		PyErr_SetString(PyExc_RuntimeError,
			"method.__self__ not accessible in restricted mode");
		return NULL;
	}
	self = m->m_self;
	if (self == NULL)
		self = Py_None;
	Py_INCREF(self);
	return self;
}

static PyGetSetDef meth_getsets [] = {
	{"__doc__",  (getter)meth_get__doc__,  NULL, NULL},
	{"__name__", (getter)meth_get__name__, NULL, NULL},
	{"__self__", (getter)meth_get__self__, NULL, NULL},
	{0}
};

#define OFF(x) offsetof(PyClosureFunctionObject, x)

static PyMemberDef meth_members[] = {
	{"__module__",    T_OBJECT,     OFF(m_module),WRITE_RESTRICTED},
	{NULL}
};

static PyObject *
meth_repr(PyClosureFunctionObject *m)
{
	if (m->m_self == NULL)
		return PyString_FromFormat("<built-in function %s>",
					   m->m_ml->ml_name);
	return PyString_FromFormat("<built-in method %s of %s object at %p>",
				   m->m_ml->ml_name,
				   m->m_self->ob_type->tp_name,
				   m->m_self);
}


PyTypeObject PyClosureFunction_Type = {
	PyObject_HEAD_INIT(&PyType_Type)
    0,
	"builtin_function_or_method",
	sizeof(PyClosureFunctionObject),
	0,
	(destructor)meth_dealloc, 		/* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,			/* tp_compare */
	(reprfunc)meth_repr,			/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,			/* tp_hash */
	PyClosureFunction_Call,			/* tp_call */
	0,					/* tp_str */
	PyObject_GenericGetAttr,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,/* tp_flags */
 	0,					/* tp_doc */
 	(traverseproc)meth_traverse,		/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	0,					/* tp_iter */
	0,					/* tp_iternext */
	0,					/* tp_methods */
	meth_members,				/* tp_members */
	meth_getsets,				/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
};

/* Clear out the free list */

static int
PyClosureFunction_ClearFreeList(void)
{
	int freelist_size = numfree;
	
	while (free_list) {
		PyClosureFunctionObject *v = free_list;
		free_list = (PyClosureFunctionObject *)(v->m_self);
		PyObject_GC_Del(v);
		numfree--;
	}
	assert(numfree == 0);
	return freelist_size;
}


typedef struct {
    PyObject_HEAD
    PyTypeObject *d_type;
    PyObject *d_name;
    PyObject *d_closure;
	PyMethodDef *d_method;
} PyClosureMethodDescrObject;




/* Descriptor */
static void
descr_dealloc(PyClosureMethodDescrObject *descr)
{
	PyObject_GC_UnTrack(descr);
	Py_XDECREF(descr->d_type);
	Py_XDECREF(descr->d_name);
    Py_XDECREF(descr->d_closure);
	PyObject_GC_Del(descr);
}

static char *
descr_name(PyClosureMethodDescrObject *descr)
{
	if (descr->d_name != NULL && PyString_Check(descr->d_name))
		return PyString_AS_STRING(descr->d_name);
	else
		return "?";
}

static PyObject *
descr_repr(PyClosureMethodDescrObject *descr, char *format)
{
	return PyString_FromFormat(format, descr_name(descr),
				   descr->d_type->tp_name);
}

static PyObject *
method_repr(PyClosureMethodDescrObject *descr)
{
	return descr_repr(descr, 
			  "<method '%s' of '%s' objects>");
}


static int
descr_check(PyClosureMethodDescrObject *descr, PyObject *obj, PyObject **pres)
{
	if (obj == NULL) {
		Py_INCREF(descr);
		*pres = (PyObject *)descr;
		return 1;
	}
	if (!PyObject_TypeCheck(obj, descr->d_type)) {
		PyErr_Format(PyExc_TypeError,
			     "descriptor '%s' for '%s' objects "
			     "doesn't apply to '%s' object",
			     descr_name((PyClosureMethodDescrObject *)descr),
			     descr->d_type->tp_name,
			     obj->ob_type->tp_name);
		*pres = NULL;
		return 1;
	}
	return 0;
}

static PyObject *
classmethod_get(PyClosureMethodDescrObject *descr, PyObject *obj, PyObject *type)
{
	/* Ensure a valid type.  Class methods ignore obj. */
	if (type == NULL) {
		if (obj != NULL)
			type = (PyObject *)obj->ob_type;
		else {
			/* Wot - no type?! */
			PyErr_Format(PyExc_TypeError,
				     "descriptor '%s' for type '%s' "
				     "needs either an object or a type",
				     descr_name(descr),
				     descr->d_type->tp_name);
			return NULL;
		}
	}
	if (!PyType_Check(type)) {
		PyErr_Format(PyExc_TypeError,
			     "descriptor '%s' for type '%s' "
			     "needs a type, not a '%s' as arg 2",
			     descr_name(descr),
			     descr->d_type->tp_name,
			     type->ob_type->tp_name);
		return NULL;
	}
	if (!PyType_IsSubtype((PyTypeObject *)type, descr->d_type)) {
		PyErr_Format(PyExc_TypeError,
			     "descriptor '%s' for type '%s' "
			     "doesn't apply to type '%s'",
			     descr_name(descr),
			     descr->d_type->tp_name,
			     ((PyTypeObject *)type)->tp_name);
		return NULL;
	}
	return PyClosureFunction_New(descr->d_method, type,NULL,descr->d_closure);
}

static PyObject *
method_get(PyClosureMethodDescrObject *descr, PyObject *obj, PyObject *type)
{
	PyObject *res;

	if (descr_check(descr, obj, &res))
		return res;
	return PyClosureFunction_New(descr->d_method, obj, NULL, descr->d_closure);
}


static PyObject *
methoddescr_call(PyClosureMethodDescrObject *descr, PyObject *args, PyObject *kwds)
{
	Py_ssize_t argc;
	PyObject *self, *func, *result;

	/* Make sure that the first argument is acceptable as 'self' */
	assert(PyTuple_Check(args));
	argc = PyTuple_GET_SIZE(args);
	if (argc < 1) {
		PyErr_Format(PyExc_TypeError,
			     "descriptor '%.300s' of '%.100s' "
			     "object needs an argument",
			     descr_name(descr),
			     descr->d_type->tp_name);
		return NULL;
	}
	self = PyTuple_GET_ITEM(args, 0);
	if (!PyObject_IsInstance(self, (PyObject *)(descr->d_type))) {
		PyErr_Format(PyExc_TypeError,
			     "descriptor '%.200s' "
			     "requires a '%.100s' object "
			     "but received a '%.100s'",
			     descr_name(descr),
			     descr->d_type->tp_name,
			     self->ob_type->tp_name);
		return NULL;
	}

	func = PyClosureFunction_New(descr->d_method, self, NULL, descr->d_closure);
	if (func == NULL)
		return NULL;
	args = PyTuple_GetSlice(args, 1, argc);
	if (args == NULL) {
		Py_DECREF(func);
		return NULL;
	}
	result = PyEval_CallObjectWithKeywords(func, args, kwds);
	Py_DECREF(args);
	Py_DECREF(func);
	return result;
}

static PyObject *
classmethoddescr_call(PyClosureMethodDescrObject *descr, PyObject *args,
		      PyObject *kwds)
{
	PyObject *func, *result;

	func = PyClosureFunction_New(descr->d_method, (PyObject *)descr->d_type, NULL, descr->d_closure);
	if (func == NULL)
		return NULL;

	result = PyEval_CallObjectWithKeywords(func, args, kwds);
	Py_DECREF(func);
	return result;
}


static PyObject *
method_get_doc(PyClosureMethodDescrObject *descr, void *closure)
{
	if (descr->d_method->ml_doc == NULL) {
		Py_INCREF(Py_None);
		return Py_None;
	}
	return PyString_FromString(descr->d_method->ml_doc);
}

static int
descr_traverse(PyObject *self, visitproc visit, void *arg)
{
	PyClosureMethodDescrObject *descr = (PyClosureMethodDescrObject *)self;
	Py_VISIT(descr->d_type);
    Py_VISIT(descr->d_closure);
	return 0;
}

static PyMemberDef descr_members[] = {
	{"__objclass__", T_OBJECT, offsetof(PyClosureMethodDescrObject, d_type), READONLY},
	{"__name__", T_OBJECT, offsetof(PyClosureMethodDescrObject, d_name), READONLY},
	{"__closure__", T_OBJECT, offsetof(PyClosureMethodDescrObject, d_closure), READONLY},
	{0}
};

static PyGetSetDef method_getset[] = {
	{"__doc__", (getter)method_get_doc},
	{0}
};

static PyTypeObject PyClosureMethodDescr_Type = {
	PyObject_HEAD_INIT(&PyType_Type)
    0,
	"closure method_descriptor",
	sizeof(PyClosureMethodDescrObject),
	0,
	(destructor)descr_dealloc,		/* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	(reprfunc)method_repr,			/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	(ternaryfunc)methoddescr_call,		/* tp_call */
	0,					/* tp_str */
	PyObject_GenericGetAttr,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags */
	0,					/* tp_doc */
	descr_traverse,				/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	0,					/* tp_iter */
	0,					/* tp_iternext */
	0,					/* tp_methods */
	descr_members,				/* tp_members */
	method_getset,				/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	(descrgetfunc)method_get,		/* tp_descr_get */
	0,					/* tp_descr_set */
};

/* This is for METH_CLASS in C, not for "f = classmethod(f)" in Python! */
static PyTypeObject PyClosureClassMethodDescr_Type = {
	PyObject_HEAD_INIT(&PyType_Type)
    0,
	"closure classmethod_descriptor",
	sizeof(PyClosureMethodDescrObject),
	0,
	(destructor)descr_dealloc,		/* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	(reprfunc)method_repr,			/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	(ternaryfunc)classmethoddescr_call,	/* tp_call */
	0,					/* tp_str */
	PyObject_GenericGetAttr,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags */
	0,					/* tp_doc */
	descr_traverse,				/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	0,					/* tp_iter */
	0,					/* tp_iternext */
	0,					/* tp_methods */
	descr_members,				/* tp_members */
	method_getset,				/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	(descrgetfunc)classmethod_get,		/* tp_descr_get */
	0,					/* tp_descr_set */
};


static PyClosureMethodDescrObject *
descr_new(PyTypeObject *descrtype, PyTypeObject *type, const char *name)
{
	PyClosureMethodDescrObject *descr;

	descr = (PyClosureMethodDescrObject *)PyType_GenericAlloc(descrtype, 0);
	if (descr != NULL) {
		Py_XINCREF(type);
		descr->d_type = type;
		descr->d_name = PyString_InternFromString(name);
		if (descr->d_name == NULL) {
			Py_DECREF(descr);
			descr = NULL;
		}
	}
	return descr;
}

static PyObject *
PyDescr_NewClosureMethod(PyTypeObject *type, PyMethodDef *method, PyObject *closure)
{
	PyClosureMethodDescrObject *descr;

	descr = (PyClosureMethodDescrObject *)descr_new(&PyClosureMethodDescr_Type,
						 type, method->ml_name);
	if (descr != NULL)
    {
		descr->d_method = method;
        Py_XINCREF(closure);
        descr->d_closure = closure;
    }
	return (PyObject *)descr;
}

static PyObject *
PyDescr_NewClosureClassMethod(PyTypeObject *type, PyMethodDef *method, PyObject *closure)
{
	PyClosureMethodDescrObject *descr;

	descr = (PyClosureMethodDescrObject *)descr_new(&PyClosureClassMethodDescr_Type,
						 type, method->ml_name);
	if (descr != NULL)
    {
		descr->d_method = method;
        Py_XINCREF(closure);
        descr->d_closure = closure;
    }
	return (PyObject *)descr;
}


static void
PyDWBaseObject_dealloc(PyDWBaseObject* self)
{
    if(self->obj_weak != NULL)
        PyObject_ClearWeakRefs((PyObject *)self);
    Py_XDECREF(self->obj_dict);
    self->ob_type->tp_free((PyObject*)self);
}


PyTypeObject PyDWBaseObject_Type = {
	PyObject_HEAD_INIT(NULL)
    0,
	"Object with built-in dict/weak pointer",
	sizeof(PyDWBaseObject),
	0,
	(destructor)PyDWBaseObject_dealloc, 		/* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,			/* tp_compare */
	0,			/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,			/* tp_hash */
	0,			/* tp_call */
	0,					/* tp_str */
	0,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,/* tp_flags */
 	0,					/* tp_doc */
 	0,		/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	offsetof(PyDWBaseObject,obj_weak),					/* tp_weaklistoffset */
	0,					/* tp_iter */
	0,					/* tp_iternext */
	0,					/* tp_methods */
	0,				/* tp_members */
	0,				/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
    0,                     /* tp_descr_get */
    0,                         /* tp_descr_set */
    offsetof(PyDWBaseObject,obj_dict),                         /* tp_dictoffset */
    0,                    /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};









static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
initclosure(void)
{
    PyObject *m;
    static void *PyClosure_API[PyClosure_API_pointers];
    PyObject *c_api_object;

    m = Py_InitModule("closure", module_methods);
    if (m == NULL)
        return;
    
    PyDWBaseObject_Type.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyDWBaseObject_Type) < 0)
        return;

    /* Initialize the C API pointer array */
    PyClosure_API[PyDescr_NewClosureMethod_NUM] = (void *)PyDescr_NewClosureMethod;
    PyClosure_API[PyDescr_NewClosureClassMethod_NUM] = (void *)PyDescr_NewClosureClassMethod;
    PyClosure_API[PyClosureFunction_New_NUM] = (void *)PyClosureFunction_New;
    PyClosure_API[PyDWBaseObject_Type_NUM] = (void *)&PyDWBaseObject_Type;

    /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)PyClosure_API, NULL);

    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);
}




