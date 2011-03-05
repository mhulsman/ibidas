from thirdparty.spark import GenericScanner, GenericParser

class Token(object):#{{{
    def __init__(self, type, attr=None):
        self.type = type
        self.attr = attr

    def __cmp__(self,o):
        return cmp(self.type,o)
    
    def __getitem__(self,pos):
        raise IndexError

    def __len__(self):
        return 0

    def __repr__(self):
        if(self.attr is None):
            return str(self.type)
        else:
            return str(self.type) + ":" + str(self.attr)[:20]#}}}

class AST(object):#{{{
    def __init__(self, type, kids=tuple()):
        self.type = type
        self.kids = kids

    def __getitem__(self,pos):
        return self.kids[pos]

    def __len__(self):
        return len(self.kids)

    def __repr__(self):
        return str(self.type) + str(self.kids)#}}}

    def __cmp__(self,o):
        return cmp(self.type,o)


class GenericASTRewriter:#{{{
    def typestring(self, node):
        return node.type

    def preorder(self, node=None):
        if not isinstance(node, (AST,Token)):
            return node
        name = 'n_' + self.typestring(node)
        if hasattr(self, name):
            func = getattr(self, name)
            node = func(node)
        else:
            node = self.default(node)

        node.kids = [self.preorder(kid) for kid in node]

        name = name + '_exit'
        if hasattr(self, name):
            func = getattr(self, name)
            node = func(node)
        return node

    def postorder(self, node, context=None):
        if not isinstance(node, (AST,Token)):
            return node
        name = 'x_' + self.typestring(node)
        if hasattr(self, name):
            func = getattr(self, name)
            node = func(node, context)
            return node

        node.kids = [self.postorder(kid,context) for kid in node]

        name = 'n_' + self.typestring(node)
        if hasattr(self, name):
            func = getattr(self, name)
            if context is None:
                node = func(node)
            else:
                node = func(node, context)
        else:
            node = self.default(node)
        return node

    def default(self, node):
        return node#}}}

