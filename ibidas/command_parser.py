from utils import util, context, infix
from parser_objs import *
import inspect
import representor
allowed_clss = (representor.Representor, context.Context, infix.Infix, int, long, float, str)


class CommandLineScanner(GenericScanner):#{{{
    def tokenize(self, input):
        self.rv = []
        GenericScanner.tokenize(self,input)
        return self.rv

    def t_anumber_0(self,s):
        r" \d*\.\d+([eE][\+\-]\d+)?[jJ]? | \d+\.([eE][\+\-]\d+)?[jJ]? | \d+[eE][\+\-]\d+[jJ]? "
        t = Token(type="float",attr=s)
        self.rv.append(t)
   
    def t_anumber_1(self,s):
        """ 
            0[bB][01]+[lL]? | 0[xX][0-9a-fA-F]+[lL]? | 0[oO][0-7]*[lL]? | \d+[lL]? 
        """
        t = Token(type="integer",attr=s)
        self.rv.append(t)

    def t_bkeywords(self, s):
        r' and[^a-zA-Z\d]+ | as[^a-zA-Z\d]+ | assert[^a-zA-Z\d]+  | break[^a-zA-Z\d]+  | class[^a-zA-Z\d]+  | continue[^a-zA-Z\d]+  | def[^a-zA-Z\d]+  | del[^a-zA-Z\d]+  | elif[^a-zA-Z\d]+  | else[^a-zA-Z\d]+  | except[^a-zA-Z\d]+  | exec[^a-zA-Z\d]+  | finally[^a-zA-Z\d]+  | for[^a-zA-Z\d]+  | from[^a-zA-Z\d]+  | global[^a-zA-Z\d]+  | if[^a-zA-Z\d]+  | import[^a-zA-Z\d]+  | in[^a-zA-Z\d]+  | is[^a-zA-Z\d]+  | lambda[^a-zA-Z\d]+  | not[^a-zA-Z\d]+  | or[^a-zA-Z\d]+  | pass[^a-zA-Z\d]+  | print[^a-zA-Z\d]+  | raise[^a-zA-Z\d]+  | return[^a-zA-Z\d]+  | try[^a-zA-Z\d]+  | while[^a-zA-Z\d]+  | with[^a-zA-Z\d]+  | yield[^a-zA-Z\d]+  '
        t = Token(type=s)
        self.rv.append(t)

    def t_cidentifier(self,s):
        r' [a-zA-Z_][a-zA-Z_\d]* '
        t = Token(type='identifier',attr=s)
        self.rv.append(t)

    def t_dsymbol_0(self,s):
        r' \"\"\" | \+\= | \-\= | \*\= | \/\= | \/\/\= | \%\= | \&\= | \|\= | \^\= | \>\>\= | \<\<\= | \*\*\= '
        t = Token(type=s)
        self.rv.append(t)

    def t_esymbol_1(self, s):
        r' \+ | \- | \*\* | \* | \/\/ | \/ | \% | \<\< | \>\> | \& | \| | \^ | \~ | \< | \> | \<\= | \>\= | \=\= | \!\= | \<\> '
        t = Token(type=s)
        self.rv.append(t)

    def t_fsymbol_2(self,s):
        r' \( | \) | \[ | \] | \{ | \} | \@ | \, | \: | \. | \` | \= | \; | \" | \''
        t = Token(type=s)
        self.rv.append(t)#}}}

    def t_gwhitespace(self,s):
        r' [\t\s\n]+ '
        t = Token(type="whitespace",attr=s)
        self.rv.append(t)
        pass

def merge_tokens(tokens):
    res = []
    for token in tokens:
        if token.attr is None:
            res.append(str(token.type))
        else:
            res.append(str(token.attr))
    return "".join(res)

def process_tokens(tokens):
    if not tokens:
        return tokens
    pos = 0
    while pos < len(tokens):
        token = tokens[pos]
        if pos < len(tokens) - 1:
            nexttoken = tokens[pos + 1]
        else:
            nexttoken = None
       
        if token == '"' or token.type == '"""' or token.type == "'":
            try:
                endpos = tokens.index(token.type,pos+1)
                ntoken = Token(type="string", attr=merge_tokens(tokens[(pos + 1):endpos]))
                tokens = tokens[:pos] + [ntoken] + tokens[(endpos + 1):]
            except ValueError:
                tokens[pos] = Token(type="incomplete_string", attr=merge_tokens(tokens[(pos + 1):]))
                tokens = tokens[:(pos + 1)]
        elif token == "whitespace":
            del tokens[pos]
            pos -= 1
        pos = pos + 1
    return tokens

class CommandLineParser(GenericParser):#{{{
    def __init__(self, start="simple_stmt"):
        GenericParser.__init__(self, start)

    def p_atom_0(self,args):
        """ 
            atom ::= identifier
            atom ::= literal
            atom ::= enclosure
        """
        return AST(type="atom",kids=args[:1])
        
    def p_enclosure_0(self,args):
        """
            enclosure ::= parenth_form
            enclosure ::= list_display
            enclosure ::= generator_expression
            enclosure ::= dict_display
            enclosure ::= set_display
            enclosure ::= string_conversion
            enclosure ::= yield_atom
        """
        return AST(type="enclosure",kids=args[:1])

    def p_literal_0(self,args):
        """
           literal ::= integer
           literal ::= string
           literal ::= incomplete_string
           literal ::= float
        """
        return AST(type="literal",kids=args[:1])

    def p_parenth_form_0(self,args):
        """
            parenth_form ::= ( expression_list )
            parenth_form ::= ( )
        """
        return AST(type="parenth_form",kids=args[1:-1])


    def p_list_display_0(self,args):
        """
            list_display ::= [ ]
            list_display ::= [ expression_list ]
            list_display ::= [ list_comprehension ] 
        """
        return AST(type="list_display",kids=args[1:-1])
   
    def p_list_comprehension(self, args):
        r" list_comprehension ::= expression list_for "
        return AST(type="list_comprehension",kids=args)

    def p_list_for(self, args):
        """
            list_for ::= for target_list in old_expression_list
            list_for ::= for target_list in old_expression_list list_iter
        """
        return AST(type="list_for",kids=args[1:2] + args[3:])
           

    def p_old_expression_list(self,args):
        """
            old_expression_list ::= old_expression 
            old_expression_list ::= old_expression_list , old_expression
        """
        return AST(type="old_expression_list",kids=args[:1] + args[2:])

    def p_old_expression(self,args):
        """
            old_expression ::= or_test
            old_expression ::= old_lambda_form
        """
        return AST(type="old_expression",kids=args[:1])


    def p_list_iter(self,args):
        """
            list_iter ::= list_for
            list_iter ::= list_if
        """
        return AST(type="list_iter",kids=args[:1])

    def p_list_if(self,args):
        """
            list_if ::= if old_expression
            list_if ::= if old_expression list_iter
        """
        return AST(type="list_if",kids=args[1:])

    def p_comprehension(self,args):
        r" comprehension ::= expression comp_for "
        return AST(type="comprehension",kids=args)

    def p_comp_for(self,args):
        """
            comp_for ::= for target_list in or_test
            comp_for ::= for target_list in or_test comp_iter
        """
        return AST(type="comp_for",kids=args[1:2] + args[3:])

    def p_comp_iter(self, args):
        """
            comp_iter ::= comp_for
            comp_iter ::= comp_if
        """
        return AST(type="comp_iter",kids=args[:1])

    def p_comp_if(self,args):
        """
            comp_if ::= if expression_nocond
            comp_if ::= if expression_nocond comp_iter
        """ 
        return AST(type="comp_if",kids=args[1:])


    def p_generator_expression(self,args):
        r" generator_expression ::= ( expression comp_for ) "
        return AST(type="generator_expression",kids=args[1:3])

    def p_string_conversion_0(self,args):
        """
            string_conversion ::= ` expression_list `
        """
        return AST(type="string_conversion",kids=args[1:-1])

    def p_primary(self,args):
        """
            primary ::= atom
            primary ::= attributeref
            primary ::= subscription
            primary ::= slicing
            primary ::= partial_slicing
            primary ::= call
        """
        return AST(type="primary",kids=args[:1])

    def p_attributeref(self,args):
        """
            primary ::= primary . identifier
            primary ::= primary .
        """
        return AST(type="attributeref",kids=args[:1] + args[2:])

    def p_subscription(self,args):
        """
            subscription ::= primary [ expression_list ] 
        """
        return AST(type="subscription",kids=args[:1] + args[2:3])


    def p_partial_slicing(self,args):
        """
             partial_slicing ::= primary [ slice_list
        """
        return AST(type="partial_slicing",kids=args[:1] + args[2:3])

    def p_slicing(self, args):
        """
            slicing ::= simple_slicing
            slicing ::= extended_slicing
        """
        return AST(type="slicing",kids=args[:1])

    def p_simple_slicing(self,args):
        """
            simple_slicing ::= primary [ short_slice ] 
        """
        return AST(type="simple_slicing",kids=args[:1] + args[2:3])

    def p_extended_slicing(self, args):
        """
            extended_slicing ::= primary [ slice_list ]
        """
        return AST(type="extended_slicing",kids=args[:1] + args[2:3])

    def p_slice_list(self, args):
        """
            slice_list ::= slice_item
            slice_list ::= slice_list , slice_item
        """
        return AST(type="slice_list",kids=args[:1] + args[2:])

    def p_slice_item(self, args):
        """
            slice_item ::= expression
            slice_item ::= proper_slice
            slice_item ::= ellipsis
        """
        return AST(type="slice_item",kids=args[:1])

    def p_proper_slice(self, args):
        """
            proper_slice ::= short_slice
            proper_slice ::= long_slice
        """
        return AST(type="proper_slice",kids=args[:1])

    def p_short_slice(self, args):
        """
            short_slice ::= :
            short_slice ::= expression :
            short_slice ::= : expression
            short_slice ::= expression : expression
        """
        none = Token(type="object",attr=None)
        if(len(args) == 1):
            kids = [none, none]
        elif(args[0] == "expression"):
            if len(args) > 2:
                kids = [args[0], args[2]]
            else:
                kids = [args[0], none]
        else:
            kids = [none, args[1]]
        return AST(type="short_slice",kids=kids)
        

    def p_long_slice(self, args):
        """
            long_slice ::= short_slice :
            long_slice ::= short_slice : expression
        """
        return AST(type="extended_slicing",kids=args[:1] + args[2:])

    def p_ellipsis(self, args):
        """
            ellipsis ::= . . .
        """
        return Token(type ="elllipsis")

    

    def p_call(self,args):
        """
            call ::= primary ( argument_list )
        """
        #FIXME
        return AST(type="call",kids=args[:1] + args[2:3])


    def p_argument_list(self, args):
        """
            argument_list ::= positional_arguments
            argument_list ::= positional_arguments , keyword_arguments
            argument_list ::= keyword_arguments
        """
        #FIXME
        return AST(type="argument_list",kids=args[:1] + args[2:])

    def p_positional_arguments(self, args):
        """
            positional_arguments ::= expression
            positional_arguments ::= positional_arguments , expression
        """
        return AST(type="positional_arguments",kids=args[:1] + args[2:])

    def p_keyword_arguments(self, args):
        """
            keyword_arguments ::= keyword_item
            keyword_arguments ::= keyword_arguments , keyword_item
        """
        return AST(type="keyword_arguments",kids=args[:1] + args[2:])

    def p_keyword_item(self, args):
        r" keyword_item ::= identifier = expression "
        return AST(type="keyword_item",kids=args[:1] + args[2:])

    def p_conditional_expression(self, args):
        """
            conditional_expression ::= or_test if or_test else expression
        """
        if(len(args) > 1):
            return AST(type="conditional_expression",kids=args[:1] + args[3:4] + args[5:])
        else:
            return AST(type="conditional_expression",kids=args[:1])


    def p_expression(self,args):
        """
            expression ::= conditional_expression
            expression ::= lambda_form
        """
        return AST(type="expression",kids=args[:1])


    def p_power(self,args):
        """
            power ::= primary
            power ::= primary ** u_expr
        """
        return AST(type="power",kids=args[:1] + args[2:])

    def p_u_expr(self, args):
        """
            u_expr ::= power
            u_expr ::= - u_expr
            u_expr ::= + u_expr
            u_expr ::= ~ u_expr
        """
        return AST(type="u_expr",kids=args)

    def p_m_expr(self, args):
        """
            m_expr ::= u_expr
            m_expr ::= m_expr * u_expr
            m_expr ::= m_expr // u_expr
            m_expr ::= m_expr / u_expr
            m_expr ::= m_expr % u_expr
        """
        return AST(type="m_expr",kids=args)

    def p_a_expr(self, args):
        """
            a_expr ::= m_expr
            a_expr ::= a_expr + m_expr
            a_expr ::= a_expr - m_expr
        """
        return AST(type="a_expr",kids=args)

    def p_shift_expr(self, args):
        """
            shift_expr ::= a_expr
            shift_expr ::= shift_expr << a_expr
            shift_expr ::= shift_expr >> a_expr
        """
        return AST(type="shift_expr",kids=args)


    def p_and_expr(self, args):
        """
            and_expr ::= shift_expr
            and_expr ::= and_expr & shift_expr
        """
        return AST(type="and_expr",kids=args)

    def p_xor_expr(self, args):
        """
            xor_expr ::= and_expr
            xor_expr ::= xor_expr & and_expr
        """
        return AST(type="xor_expr",kids=args)

    def p_or_expr(self, args):
        """
            or_expr ::= xor_expr
            or_expr ::= or_expr | xor_expr
        """
        return AST(type="or_expr",kids=args)


    def p_comparision(self, args):
        """
            comparison ::= or_expr
            comparison ::= comparison < or_expr
            comparison ::= comparison > or_expr
            comparison ::= comparison == or_expr
            comparison ::= comparison >= or_expr
            comparison ::= comparison <= or_expr
            comparison ::= comparison <> or_expr
            comparison ::= comparison != or_expr
            comparison ::= comparison is or_expr
            comparison ::= comparison is not or_expr
            comparison ::= comparison in or_expr
            comparison ::= comparison not in or_expr
        """
        return AST(type="comparison",kids=args)

    def p_not_test(self, args):
        """
            not_test ::= comparison
            not_test ::= not not_test
        """
        return AST(type="not_test",kids=args[:1] + args[2:])

    def p_and_test(self, args):
        """
            and_test ::= not_test
            and_test ::= and_test and not_test
        """
        return AST(type="and_test",kids=args[:1] + args[2:])

    def p_or_test(self, args):
        """
            or_test ::= and_test
            or_test ::= or_test or and_test
        """
        return AST(type="or_test",kids=args[:1] + args[2:])

    def p_conditional_expression(self, args):
        """
            conditional_expression ::= or_test 
            conditional_expression ::= or_test if or_test else expression
        """
        return AST(type="conditional_expression",kids=args[:1] + args[2:3] + args[4:])

    def p_expression(self, args):
        """
            expression ::= conditional_expression
            expression ::= lambda_form
        """
        return AST(type="expression",kids=args[:1])

    def p_lambda_form(self,args):
        """
            lambda_form ::= lambda : expression 
            lambda_form ::= lambda parameter_list : expression
        """
        if len (args) == 3:
            return AST(type="lambda_form",kids=args[2:])
        else:
            return AST(type="lambda_form",kids=args[1:2] + args[3:])

    def p_old_lambda_form(self,args):
        """
            old_lambda_form ::= lambda : old_expression 
            old_lambda_form ::= lambda parameter_list : old_expression
        """
        if len (args) == 3:
            return AST(type="old_lambda_form",kids=args[2:])
        else:
            return AST(type="old_lambda_form",kids=args[1:2] + args[3:])


    def p_expression_list(self, args):
        """
            expression_list ::= expression
            expression_list ::= expression_list , expression
        """
        return AST(type="expression_list",kids=args[:1] + args[2:])

    def p_simple_stmt(self, args):
        """
            simple_stmt ::= expression_stmt
            simple_stmt ::= assert_stmt
            simple_stmt ::= assignment_stmt
            simple_stmt ::= augmented_assignment_stmt
            simple_stmt ::= pass_stmt
            simple_stmt ::= del_stmt
            simple_stmt ::= print_stmt
            simple_stmt ::= return_stmt
            simple_stmt ::= yield_stmt
            simple_stmt ::= raise_stmt
            simple_stmt ::= break_stmt
            simple_stmt ::= continue_stmt
            simple_stmt ::= import_stmt
            simple_stmt ::= global_stmt
            simple_stmt ::= exec_stmt
        """
        return AST(type="simple_stmt",kids=args[:1])

    def p_expression_stmt(self, args):
        """
            expression_stmt ::= expression_list
        """
        return AST(type="expression_stmt", kids=args[:1])
    def p_assignemnt_stmt(self, args):
        """
            assignment_stmt ::= target_list = expression_list
            assignment_stmt ::= target_list = yield_expression
        """
        #FIXME
        return AST(type="assignment_stmt", kids=args[:1] + args[2:])
    
    def p_target_list(self, args):
        """
            target_list ::= target
            target_list ::= target_list , target
        """
        return AST(type="target_list", kids=args[:1] + args[2:])

    def p_target(self, args):
        """
            target ::= identifier
            target ::= ( target_list )
            target ::= [ target_list ] 
            target ::= attributeref
            target ::= subscription
            target ::= slicing
        """
        if(len(args) == 1):
            kids = args
        else:
            kids = args[1:2]
        return AST(type="target", kids=kids)
        

#}}}

class CommandLineASTRewriterPass1(GenericASTRewriter):#{{{
    def process(self, tree):
        ntree = self.postorder(tree)
        return ntree
    
    def n_slice_list(self, node):
        if(node.kids[0] == "slice_list"):
            node.kids = node.kids[0].kids + node.kids[1:]
        return node

    def n_positional_arguments(self, node):
        if(node.kids[0] == "positional_arguments"):
            node.kids = node.kids[0].kids + node.kids[1:]
        return node
    
    def n_keyword_arguments(self, node):
        if(node.kids[0] == "keyword_arguments"):
            node.kids = node.kids[0].kids + node.kids[1:]
        return node
   
    def n_target_list(self, node):
        if(node.kids[0] == "target_list"):
            node.kids = node.kids[0].kids + node.kids[1:]
        return node
    
    def n_expression_list(self, node):
        if(node.kids[0] == "expression_list"):
            node.kids = node.kids[0].kids + node.kids[1:]
        return node

class CommandLineASTRewriterPass2(GenericASTRewriter):#{{{
    def process(self, tree, ipshell):
        self.ipshell = ipshell
        ntree = self.postorder(tree)
        return ntree

    def objectify(self, node, pos, context=None):
        if node.kids[pos] == "identifier":
            if context is not None and node.kids[pos].attr == "_":
                node.kids[pos] = Token(type="object", attr= context)
            else:
                node.kids[pos] = Token(type="object", attr= get_from_namespace(node.kids[pos].attr, self.ipshell))
        return node

    def n_attributeref(self, node, context=None):
        node = self.objectify(node,0, context)
        if node.kids[0] != "object":
            return node
       
        obj = node.kids[0].attr
        if(len(node.kids) == 1):
            node.kids.append(Token(type="identifier",attr=""))
        elif node.kids[1] == "identifier" and hasattr(obj, node.kids[1].attr):
            res = getattr(obj, node.kids[1].attr)
            return Token(type="object",attr=res)

        return node

    def n_slicing(self, node, context=None):
        node.kids = node.kids[0].kids
        node = self.objectify(node,0, context)

        if node.kids[0] != "object" or not isinstance(node.kids[0].attr, allowed_clss):
            return node
        obj = node.kids[0].attr

        if node.kids[1] == "object":
            return Token(type="object",attr=obj[node.kids[1].attr])

        return node
      
    def x_partial_slicing(self, node, context=None):
        node.kids[0] = self.postorder(node.kids[0], context)
        node = self.objectify(node,0,context)

        if node.kids[0] == "object":
            context = node.kids[0].attr

        return self.postorder(node.kids[1],context)



    def n_float(self, node, context=None):
        return Token(type="object", attr=float(node.attr))

    def n_integer(self, node, context=None):
        if node.attr[-1] == 'l' or node.attr[-1] == 'L':
            return Token(type="object", attr=long(node.attr))
        else:
            return Token(type="object", attr=int(node.attr))

    def n_string(self, node, context=None):
        return Token(type="object", attr=node.attr)

    def n_short_slice(self, node, context):
        if all([kid == "object" for kid in node.kids]):
            return Token(type="object", attr=slice(*[kid.attr for kid in node.kids]))
        return node

    def n_slice_list(self, node, context=None):
        if(len(node.kids) == 1):
            return node.kids[0]
        if not all([kid == "object" for kid in node.kids]):
            return node

        return Token(type="object", attr= tuple([kid.attr for kid in node.kids]))

    def n_target_list(self, node, context=None):
        if(len(node.kids) == 1):
            return node.kids[0]
        return node

    def n_argument_list(self, node, context=None):
        nkids = []
        params = []
        keywords = {}
        for kid in node.kids:
            if kid == 'positional_arguments':
                if(all([skid == "object" for skid in kid.kids])):
                    params.extend([skid.attr for skid in kid.kids])
                else:
                    return node
            elif kid == 'keyword_arguments':
                if(not all([skid == "keyword_item" for skid in kid.kids])):
                    return node
                if(not all([skid.kids[1] == "object" for skid in kid.kids])):
                    return node
                for skid in kid.kids:
                    keywords[skid.kids[0].attr] = skid.kids[1].attr
        node.keywords = keywords
        node.params = params
        return node

    def n_assignment_stmt(self, node, context=None):
        #no assignments handled yet
        return node.kids[1]

    def n_call(self, node, context=None):
        if not (node.kids[0] == "object" and node.kids[1] == "argument_list" and hasattr(node.kids[1],'params')):
            return node

        method = node.kids[0].attr
        if(inspect.ismethod(method)):
           if not isinstance(method.im_self, allowed_clss):
               return node
           if isinstance(method.im_self, representor.Representor) and method.im_func.func_name in ['__str__','__nonzero__','Copy','__reduce__']:
               return node
           res = method(*node.kids[1].params, **node.kids[1].keywords)
           return Token(type="object", attr= res)


    def binoperator(self, node, context=None):
        if(len(node.kids) == 1):
            return node.kids[0]
        node = self.objectify(node,0, context)
        if node.kids[0] =="object" and node.kids[2] == "object":
            obj1 = node.kids[0].attr
            obj2 = node.kids[2].attr
            if(isinstance(obj1, allowed_clss) and isinstance(obj2, allowed_clss)):
                res = eval('obj1 ' + node.kids[1].type + ' obj2')
                return Token(type="object", attr= res)
        return node
    n_comparison = binoperator
    n_or_expr = binoperator
    n_and_expr = binoperator
    n_xor_expr = binoperator
    n_shift_expr = binoperator
    n_a_expr = binoperator
    n_m_expr = binoperator
    n_power = binoperator



    def n_removenode(self, node, context=None):
        if len(node.kids) == 1:
            return node.kids[0]
        return node
   
    n_expression_list = n_removenode
    n_expression = n_removenode
    n_conditional_expression = n_removenode
    n_and_test = n_removenode
    n_or_test = n_removenode
    n_not_test = n_removenode
    n_u_expr = n_removenode
    n_literal = n_removenode
    n_enclosure = n_removenode
    n_atom = n_removenode
    n_primary = n_removenode
    
    n_proper_slice = n_removenode
    n_slice_item = n_removenode
    n_simple_stmt = n_removenode
    n_expression_stmt = n_removenode
#}}}

class CommandLineASTInterpreterPass(GenericASTRewriter):#{{{
    def process(self, tree):
        ntree = self.toponly(tree)
        return ntree

    def toponly(self, node=None):
        name = 'n_' + self.typestring(node)
        if hasattr(self, name):
            func = getattr(self, name)
            node = func(node)
        else:
            node = self.default(node)
        return node

    def n_attributeref(self, node):
        if node.kids[0] == "object" and node.kids[1] == "identifier":
            return (node.kids[0].attr, node.kids[1].attr) 
        return node

    

#}}}


def get_from_namespace(attr, ipshell):
    if attr in ipshell.IP.user_ns:
        return ipshell.IP.user_ns[attr]
    else:
        return ipshell.IP.user_global_ns[attr]

def parseCommandLine(line, ipshell, cursor_pos=None, debug=False):
    if cursor_pos is None:
        parse_line = line
    else:
        parse_line = line[:cursor_pos]
    scanner = CommandLineScanner()
    tokens = scanner.tokenize(parse_line)
    if(debug):
        print 1, tokens
    tokens = process_tokens(tokens)
    if(debug):
        print 2, tokens
    parser = CommandLineParser()
    tree = parser.parse(tokens)
    if(debug):
        print 3, tree
    
    rewriter = CommandLineASTRewriterPass1()
    tree = rewriter.process(tree)
    if(debug):
        print 4, tree
   
    rewriter = CommandLineASTRewriterPass2()
    tree = rewriter.process(tree,ipshell)
    if(debug):
        print 5, tree

    interpreter = CommandLineASTInterpreterPass()
    res = interpreter.process(tree)
    if(debug):
        print 6, res
    return res
