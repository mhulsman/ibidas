import __builtin__

delayed_imports = []

def delay_import(module, name, *args):
    global delayed_imports
    delayed_imports.append((module, name, args))

def perform_delayed_imports():
    global delayed_imports
    delayed_imports = delayed_imports[::-1]
    while delayed_imports:
        module, name, args = delayed_imports.pop()
        ximport(name,module,args)


__builtin__.__dict__['_delay_import_'] = delay_import

def ximport(name, module, args=[]):
    level = 0
    while(name.startswith('.')):
        name = name[1:]
        level += 1
    
    if(level == 0):
        level = -1
       
    try: 
        if(not args):
            module[name.split('.')[0]] = __import__(name,module,module,[],level)
        elif(len(args) == 1 and args[0] == "*"):
            res = __import__(name,module,module,[],level)
            for arg in res.__dict__.keys():
                if(arg.startswith('__')):
                    continue
                try:
                    module[arg] = getattr(res,arg)
                except AttributeError,e:
                    raise RuntimeError,"importing " + str(arg) + " from " + name + " into " + module['__name__'] + " did not succeed"
        else:
            try:
                res = __import__(name,module,module,args,level)
                for arg in args:
                    try:
                        module[arg] = getattr(res,arg)
                    except AttributeError,e:
                        raise RuntimeError,"importing " + str(arg) + " from " + name + " into " + module['__name__'] + " did not succeed"
            except ValueError:
                for arg in args:
                    res = __import__(name + "." + arg,module,module,args,level)
                    module[arg] = res
    except ImportError, e:
        raise ImportError("importing from " + name + "(" + str(level) + "): " + str(args) + " into " + module["__name__"] + " did not succeed: " + e.message)
