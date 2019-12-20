import importlib

def find_class(module, name):
    if name.rfind('.') != -1:
        m = name[:name.rfind('.')]
        c = name[name.rfind('.') + 1:]
        print("load class from ", module, m, "C", c)
        try:
            if module is not None:
                ret = getattr(importlib.import_module(module, m), c)
            else:
                ret = getattr(importlib.import_module(m), m)
        except AttributeError as e:
            ret = None

        if ret is None:
            print("failover load class from ", m, "C", c)
            ret = getattr(importlib.import_module(m), c)
    else:
        print("load class from ", module, name)
        ret = getattr(importlib.import_module(module), name)

    return ret

class dict2obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)

def create_dataset(type_class, params):
    return type_class(**params)