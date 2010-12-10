#ifndef Py_MVISITOR_H
#define Py_MVISITOR_H

#ifdef __cplusplus
extern "C" {
#endif



#define Visitor_factory_NUM 0
#define Visitor_factory_RETURN PyObject *
#define Visitor_factory_PROTO (PyObject *, PyObject *, PyObject *, PyObject *)

#define Visitor_visitkey_NUM 1
#define Visitor_visitkey_RETURN PyObject *
#define Visitor_visitkey_PROTO (PyObject *, PyObject **)

#define Visitor_findmethod_NUM 2
#define Visitor_findmethod_RETURN PyObject *
#define Visitor_findmethod_PROTO (PyObject *, PyObject *,PyObject *)

#define Visitor_visitify_NUM 3
#define Visitor_visitify_RETURN int
#define Visitor_visitify_PROTO (PyObject *, PyTypeObject *,PyObject *, PyObject *)

#define Visitor_notfound_NUM 4
#define Visitor_notfound_RETURN PyObject *
#define Visitor_notfound_PROTO (PyObject *, PyObject *, PyObject *, long)

#define Visitor_visit_NUM 5
#define Visitor_visit_RETURN PyObject *
#define Visitor_visit_PROTO (PyObject *, PyObject *, PyObject *, PyObject *)


/* Total number of C API pointers */
#define PyMVisitor_API_pointers 6 


#define VISIT_NF_ELSE  0x0001
#define VISIT_NF_ERROR 0x0002
#define VISIT_NF_ROBJ  0x0004
#define VISIT_CACHE    0x0008

typedef PyObject *(*VisitKeyFunction)(PyObject *, PyObject **);
typedef PyObject *(*FindMethodFunction)(PyObject *, PyObject *, PyObject *);
typedef int (*VisitifyFunction)(PyObject *, PyTypeObject *, PyObject *, PyObject *);
typedef PyObject *(*NotFoundFunction)(PyObject *, PyObject *, PyObject *,  long);
typedef PyObject *(*VisitorFactoryFunction)(PyObject *, PyObject *, PyObject *,  PyObject *);


/* helper function to create closure object for VisitorFactory function */
static PyObject * create_closure(VisitKeyFunction vkf, FindMethodFunction fmf, 
                VisitifyFunction vf, NotFoundFunction nff)
{

    PyObject *vsfuncobj,* vkfuncobj,* fmfuncobj, *nffuncobj, *closure;
    vsfuncobj = PyCObject_FromVoidPtr(vf,NULL);
    vkfuncobj = PyCObject_FromVoidPtr(vkf,NULL);
    fmfuncobj = PyCObject_FromVoidPtr(fmf,NULL);
    nffuncobj = PyCObject_FromVoidPtr(nff,NULL);
    if(vsfuncobj == NULL || vkfuncobj == NULL || fmfuncobj == NULL
        || nffuncobj == NULL)
        return NULL;
    
    //function closure object
    closure = PyTuple_Pack(4,vsfuncobj,vkfuncobj,fmfuncobj,nffuncobj);
    Py_DECREF(vsfuncobj);
    Py_DECREF(vkfuncobj);
    Py_DECREF(fmfuncobj);
    Py_DECREF(nffuncobj);

    return closure;
}

#ifdef MVISITOR_MODULE
    static Visitor_factory_RETURN VisitorFactory Visitor_factory_PROTO;
    static Visitor_visitkey_RETURN _visitkey Visitor_visitkey_PROTO;
    static Visitor_findmethod_RETURN _findmethod Visitor_findmethod_PROTO;
    static Visitor_visitify_RETURN _visitify Visitor_visitify_PROTO;
    static Visitor_notfound_RETURN _notfound Visitor_notfound_PROTO;
    static Visitor_visit_RETURN visit Visitor_visit_PROTO;


#else
    static void **PyMVisitor_API;
    static PyObject * VisitorFactory;
    
    #define Visitor_factory \
    (*(Visitor_factory_RETURN (*)Visitor_factory_PROTO) PyMVisitor_API[Visitor_factory_NUM])
    
    #define Visitor_visitkey \
    (*(Visitor_visitkey_RETURN (*)Visitor_visitkey_PROTO) PyMVisitor_API[Visitor_visitkey_NUM])
    
    #define Visitor_findmethod \
    (*(Visitor_findmethod_RETURN (*)Visitor_findmethod_PROTO) PyMVisitor_API[Visitor_findmethod_NUM])
    
    #define Visitor_visitify \
    (*(Visitor_visitify_RETURN (*)Visitor_visitify_PROTO) PyMVisitor_API[Visitor_visitify_NUM])
    
    #define Visitor_notfound \
    (*(Visitor_notfound_RETURN (*)Visitor_notfound_PROTO) PyMVisitor_API[Visitor_notfound_NUM])

    #define Visitor_visit \
    (*(Visitor_visit_RETURN (*)Visitor_visit_PROTO) PyMVisitor_API[Visitor_visit_NUM])
    /* Return -1 and set exception on error, 0 on success. */
    static int
    import_multi_visitor(void)
    {
        PyObject *globals = PyEval_GetGlobals();
        PyObject *module = PyImport_ImportModuleEx("multi_visitor",globals,NULL,NULL);

        if (module != NULL) 
        {
            PyObject *c_api_object = PyObject_GetAttrString(module, "_C_API");
            VisitorFactory = PyObject_GetAttrString(module, "VisitorFactory");
            if (c_api_object == NULL || VisitorFactory == NULL)
                return -1;
            if (PyCObject_Check(c_api_object))
                PyMVisitor_API = (void **)PyCObject_AsVoidPtr(c_api_object);
            Py_DECREF(c_api_object);
        }
        return 0;
    }
#endif


#ifdef __cplusplus
}
#endif

#endif

