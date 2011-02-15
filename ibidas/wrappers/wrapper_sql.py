import copy
import wrapper
import operator
import sqlalchemy
from sqlalchemy.sql import expression as sql

from .. import ops
from ..constants import *
from ..utils.multi_visitor import VisitorFactory, NF_ELSE, F_CACHE, NF_ERROR
from ..utils import util

from ..passes import manager, create_graph, annotate_replinks, serialize_exec
from .. import query_graph

_delay_import_(globals(),"..utils","nested_array")
_delay_import_(globals(),"..itypes","rtypes","dimensions","dimpaths")
_delay_import_(globals(),"wrapper_py")

class TypeMapperToSQLAlchemy(VisitorFactory(prefixes=("convert",), #{{{
                                      flags=NF_ELSE)):
    
    def convertTypeAny(self, rtype):
        return sqlalchemy.PickleType()
    
    def convertTypeUInt64(self, rtype): #to be checked: can databases handle uint64?
        return sqlalchemy.types.Integer()       

    def convertTypeInt64(self, rtype):
        return sqlalchemy.types.Integer()        
    
    def convertTypeInt16(self, rtype):
        return sqlalchemy.types.SmallInteger()        
   
    def convertTypeBool(self, rtype):
        return sqlalchemy.types.Boolean()
    
    def convertTypeReal64(self, rtype):
        return sqlalchemy.types.Float()
    
    def convertTypeString(self, rtype):
        if(rtype.dims and rtype.dims[0].shape > 0):
            return sqlalchemy.types.Text(length = rtype.dims[0].shape)
        else:
            return sqlalchemy.types.Text()#}}}
tosa_typemapper = TypeMapperToSQLAlchemy()


class TypeMapperFromSQLAlchemy(VisitorFactory(prefixes=("convert",), #{{{
                                      flags=NF_ELSE)):
    
    def convertAny(self, dbtype, column, typestr="any"):
        if(column.nullable):
            typestr += "$"
        return rtypes.createType(typestr)
    convertNullType = convertAny
    
    def convertBigInteger(self, dbtype, column):
        return self.convertAny(dbtype, column, "int64")
   
    def convertInteger(self, dbtype, column):
        return self.convertAny(dbtype, column, "int32")
    
    def convertSmallInteger(self, dbtype, column):
        return self.convertAny(dbtype, column, "int16")
    
    def convertBoolean(self, dbtype, column):
        return self.convertAny(dbtype, column, "bool")
    
    def convertFloat(self, dbtype, column):
        if(dbtype.precision == 24):
            return self.convertAny(dbtype, column, "real32")
        elif(dbtype.precision == 53):
            return self.convertAny(dbtype, column, "real64")
        else:
            return self.convertAny(dbtype, column, "real64")
            #raise RuntimeError, "Unknown precision: " + str(dbtype)
    
    def convertNumeric(self, dbtype, column):
        return self.convertAny(dbtype, column, "number")
    
    def convertDateTime(self, dbtype, column):
        return self.convertAny(dbtype, column)
    
    def convertTime(self, dbtype, column):
        return self.convertAny(dbtype, column)
    
    def convertDate(self, dbtype, column):
        return self.convertAny(dbtype, column)
    
    def convertInterval(self, dbtype, column):
        return self.convertAny(dbtype, column)

    def convertUnicode(self, dbtype, column):
        return self.convertString(dbtype, column, "string")

    def convertBinary(self, dbtype, column):
        return self.convertString(dbtype, column)

    def convertString(self, dbtype, column, deftype="bytes"):
        if(dbtype.length is None or dbtype.length > 32):
            typestr = "bytes"
        else:
            typestr = "bytes[%d]" % dbtype.length
        if(column.nullable):
            typestr += "$"
        
        return rtypes.createType(typestr)

    def convertPGArray(self, dbtype, column):
        subtype = self.convert(dbtype.item_type, column)
        subtype.has_missing = False
        typestr = "[~](" + str(subtype)
        if(column.nullable):
            typestr += "$"
        typestr += ")"
        return rtypes.createType(typestr)#}}}

sa_typemapper = TypeMapperFromSQLAlchemy()

postgres_types = {}
def _getPostgresTypes(engine):#{{{
    """Returns a list of dictionaries with 
        information on each type in the database"""
    if(not engine in postgres_types):
        res = engine.execute("SELECT oid,typname,typlen,typelem,\
                                typdefault,typbasetype,typnotnull,typtype \
                        FROM pg_type");
        type_info_lists = res.fetchall()
        type_info_dicts = dict([(til[0],{
                            'name': til[1], 
                            'length':til[2], 
                            'array_element_type_index':til[3], \
                            'default_expression':til[4],
                            'base_type':til[5],\
                            'not_null':til[6], \
                            'type':til[7]}
                             ) for til in type_info_lists])
        postgres_types[engine] = type_info_dicts
    return postgres_types[engine]#}}}
pgtoibidas = {'int2':'int16','varchar':'string[]','int4':'int32','text':'string[]', 'float4':'real32', 'float8':'real64'}


mysql_types = {}
def _getMySQLTypes(engine):
    """Returns a list of dictionaries with 
        information on each type in the database"""
    return mysqlid_toibidas
mysqlid_toibidas = {0:'real64',1:'int8',2:'int16',3:'int32',4:'real32',5:'real64',6:'any',7:'any',8:'int64',9:'int32',10:'any',
                    11:'any', 12:'any',13:'uint16',14:'any',15:'string[]',16:'bool',246:'any',247:'any',248:'any',249:'any',250:'any',
                    251:'any',252:'any',253:'string[]',254:'string[]',255:'any'};

def convert(col_descriptor, type, engine, tablename=None):#{{{
    if(type == 'postgres'):
        fieldnames, subtypes = convert_postgres(col_descriptor, engine)
    elif(type == 'sqlalchemy'):
        fieldnames, subtypes = convert_sqlalchemy(col_descriptor, engine)
    elif(type == 'mysql'):
        fieldnames, subtypes = convert_mysql(col_descriptor, engine)
    else:
        raise RuntimeError, "Unimplemented dialect"
    
    
    table_type = rtypes.TypeTuple(has_missing=False, 
                                    subtypes=tuple(subtypes), 
                                    fieldnames = tuple(fieldnames))

    if(tablename):
        ndims = dimpaths.DimPath(dimensions.Dim("*", name=str(tablename).lower()))
    else:
        ndims = dimpaths.DimPath(dimensions.Dim("*"))

    table_type = rtypes.TypeArray(dims=ndims,
                                    subtypes=(table_type,))
    return table_type#}}}
        

def convert_mysql(col_descriptor, engine):#{{{
    fieldnames = []
    subtypes = []
    for col in col_descriptor:
        name, type_code, display_size, internal_size, precision, scale, null_ok = col
        r = _getMySQLTypes(engine)
        d = r[type_code]
        if(null_ok):
                d += "$"
        subtypes.append(rtypes.createType(d))
        fieldnames.append(util.valid_name(name))
    return (fieldnames, subtypes)#}}}
   

def convert_postgres(col_descriptor, engine):#{{{
    fieldnames = []
    subtypes = []
    for col in col_descriptor:
        name, type_code, display_size, internal_size, precision, scale, null_ok = col
        r = _getPostgresTypes(engine)
        n = r[type_code]
        if(n['name'] in pgtoibidas):
            d = pgtoibidas[n['name']]
        elif(n['array_element_type_index'] > 0):
            subtype_id = n['array_element_type_index']
            sn = r[subtype_id]
            if(sn['name'] in pgtoibidas):
                sd = pgtoibidas[sn['name']]
                #if(not sn['not_null']):
                #    sd += "$"
            else:
                sd = "any$"
            if(n['length'] == -1):
                d = "[~](" + sd + ")"
            else:
                d = "[" + str(n['length']) + "](" + sd + ")"
        else:
            d = "any"
        if(null_ok):
                d += "$"
        subtypes.append(rtypes.createType(d))
        fieldnames.append(util.valid_name(name))
    return (fieldnames, subtypes)#}}}

def convert_sqlalchemy(col_descriptor, engine):
    fieldnames = []
    subtypes = []
    for column in col_descriptor:
        fieldnames.append(util.valid_name(column.name))
        subtypes.append(sa_typemapper.convert(column.type, column))

    return fieldnames, subtypes

def open_db(*args, **kwargs):
    con = sqlalchemy.create_engine(*args, **kwargs)
    return Connection(con)

class Connection(object):
    def __init__(self, engine, schemaname=None):#{{{
        self.engine = engine
        self.meta = sqlalchemy.MetaData()
        if(schemaname):
            self.meta.reflect(bind=self.engine, schema=schemaname)
        else:
            self.meta.reflect(bind=self.engine)
        
        tables = self.meta.tables
        self.sa_tables = tables

        tabledict = {}
        for table in tables.values():
            t = TableRepresentor(self.engine, table).elements().attributes()
            tabledict[table.name] = t
        self.tabledict = tabledict 

        if(not schemaname):
            try:
                schemanames = self.engine.execute('SELECT schema_name FROM information_schema.schemata').fetchall()
                for sname in schemanames:
                    sname = sname[0]
                    self.__dict__[sname] = Connection(self.engine, sname);
            except:
                pass#}}}

    def __getattr__(self, name):
        return self.tabledict[name]

    def _getAttributeNames(self):
        return self.tabledict.keys()
   
    
    def query(self, query):
        return QueryRepresentor(self.engine, query).elements().attributes()

    def __repr__(self):
        res = "Database: " + str(self.engine.url) + "\n"
        tablenames = self.tabledict.keys()
        tablenames.sort()
        res += "Tables: " + ", ".join([str(tname) for tname in tablenames])
        return res


    def store(self, name, rep, primary_slices=None, indexes = {}, unique = {}):#{{{
        if(not name in self.tabledict):
            columns = []
            for slice in rep._slices:
                stype= tosa_typemapper.convert(slice.type)
                nullable = slice.type.has_missing
                columns.append(sqlalchemy.Column(slice.name, stype, nullable = nullable))
     
            newtable = sqlalchemy.Table(name, self.meta, *columns) 
            newtable.create(bind=self.engine, checkfirst=True)
     
            self.sa_tables[name] = newtable
            table = self.meta.tables[name]
            t = TableRepresentor(self.engine, table).elements().attributes()
            self.tabledict[name] = t
        
        self._append(name, rep)
        return self.tabledict[name]#}}}
    
    def _append(self, name, rep):#{{{
       w = dict([(s.name,s) for s in rep._slices])
       cols = [slice for slice in self.tabledict[name]._slices if slice.name in w]
       assert len(cols) == len(w), "Slices not in table: " + set(w) - set(colnames)
       
       for s in cols:
          if not s.type >=w[s.name].type:
              raise RuntimeError, "Incompatible types: " + str(s.type) + " " + str(w[s.name].type)
       rep = rep.get(*[slice.name for slice in cols])
       
       trep = rep.tuple()
       tbdims = set([slice.dims for slice in trep._slices])
       assert len(tbdims) == 1 and len(tbdims.pop()) == 1, \
            "Only a representor object with one common parent dimension can be \
                stored as a table"

       data = rep.to_python()
       values = [{} for elem in data]
       if(len(cols) == 1):
            sname = cols[0].name
            for pos in xrange(len(values)):
                values[pos][sname] = data[pos]
       else:
            for colnr, slice in enumerate(cols):
                sname = slice.name
                for pos in xrange(len(values)):
                    values[pos][sname] = data[pos][colnr]
       self.engine.execute(self.sa_tables[name].insert(), values)#}}}

    def drop(self, name):#{{{
        self.sa_tables[name].drop(bind=self.engine, checkfirst=True)
        del self.sa_tables[name]
        del self.tabledict[name]#}}}

class TableRepresentor(wrapper.SourceRepresentor):#{{{
    def __init__(self, engine, table):
        table_type = convert(table.columns,'sqlalchemy',engine, table.name)
        nslices = (SQLOp(engine, table, table_type, table.name),)
        self._initialize(nslices,RS_ALL_KNOWN)#}}}

class QueryRepresentor(wrapper.SourceRepresentor):#{{{
    def __init__(self, engine, query):
        res = self.engine.execute(query)
        try:
            query_type = convert(res.context.compiled.statement.columns, "sqlalchemy", self.engine)
        except AttributeError:
            query_type = convert(res.cursor.description, res.dialect.name, self.engine)
        res.close() 
        
        nslices = (SQLOp(engine, query, query_type),)
        self._initialize(nslices,RS_ALL_KNOWN)#}}}



class SQLResultEdge(query_graph.Edge):#{{{
    __slots__ = ["realedge","pos"]
    def __init__(self, source, target, realedge, fieldpos):
        self.realedge = realedge
        self.pos = fieldpos
        query_graph.Edge.__init__(self, source, target)#}}}

class SQLTransEdge(SQLResultEdge):#{{{
    __slots__ = ["targetpos"]
    def __init__(self, source, target, realedge, fieldpos, tpos):
        self.targetpos = tpos
        SQLResultEdge.__init__(self,source,target,realedge,fieldpos)#}}}
    

class SQLPlanner(VisitorFactory(prefixes=("eat","expressionEat"), 
                                      flags=NF_ELSE), manager.Pass):
    after=set([create_graph.CreateGraph, annotate_replinks.AnnotateRepLinks])
    before=set([serialize_exec.SerializeExec])

    @classmethod
    def run(cls, query, run_manager, debug_mode=False):#{{{
        self = cls()
        self.graph = run_manager.pass_results[create_graph.CreateGraph]
        self.expressions = run_manager.pass_results[annotate_replinks.AnnotateRepLinks]

        self.done = set()
        self.sql_obj = set()
        self.eat_front = self.graph.getSources()

        while(self.eat_front):
            self.next_round = set()
            if(debug_mode):
                from ..passes.cytoscape_vis import DebugVisualizer
                DebugVisualizer.run(query,run_manager)
            for node in list(self.eat_front):
                self.eat(node)
            
            self.eat_front = self.next_round

        for node in self.sql_obj:
            if(isinstance(node, SQLQuery)):
                self.finish(node)
            else:
                self.graph.dropNode(node)#}}}

    #UTILS
    def allSQLParams(self, in_slice, expression):#{{{
        edges = self.graph.edge_target[in_slice]
        paramedges = set()
        sqlparamedges = set()
        for edge in edges:
            if(edge.source in expression.all_slices):
                continue
            if isinstance(edge, query_graph.ParamEdge):
                paramedges.add(edge)
            elif isinstance(edge, SQLResultEdge):
                sqlparamedges.add(edge.realedge)
        return paramedges == sqlparamedges#}}}

    def getSQLEdge(self, edge):#{{{
        etargets = self.graph.edge_target[edge.target]
        for etarget in etargets:
            if isinstance(etarget,SQLResultEdge):
                if(etarget.realedge == edge):
                   return etarget
        return False#}}}

    def compatible(self,elem1,elem2):#{{{
        if(isinstance(elem1, SQLValue) or isinstance(elem2, SQLValue)):
            return True

        if(isinstance(elem2, SQLTerm)):
            elem1, elem2 = elem2, elem1

        if(isinstance(elem1, SQLTerm)):
            s = elem1.getSource(self.graph)
            if(isinstance(elem2, SQLQuery)):
                return s == elem2 or s is None
            elif(isinstance(elem2, SQLTerm)):
                s2 = elem2.getSource(self.graph)
                return s == s2 or s is None or s2 is None
        elif(isinstance(elem1, SQLQuery)):
            if(isinstance(elem2, SQLQuery)):
                return elem1 == elem2        
         
        raise RuntimeError, "Encountered unknown type of SQL object: " + str(elem1) + " and " + str(elem2)#}}}
            
    def addPairToGraph(self, node, node2):#{{{
        self.graph.addNode(node)
        self.graph.addNode(node2)
        self.graph.addEdge(query_graph.ParamEdge(node,node2,"slice"))#}}}

    #SOURCE OBJECTS
    def eatSQLOp(self, node):#{{{
        o = SQLQuery(node.conn)
        self.graph.addNode(o)

        self.next_round.add(o)
        self.sql_obj.add(o)

        etargets = list(self.graph.edge_source[node])
        assert len(etargets) == 1, "Expected one target slice after sql op"
        anode = etargets[0].target
        assert isinstance(anode,ops.UnpackArrayOp), "Expected array unpack operation after sql op"
        
        targets = [edge.target for edge in self.graph.edge_source[anode]]
        columns = list(node.query.columns)
        ncolumns = []
        
        for pos,target in enumerate(targets):
            assert isinstance(target,ops.UnpackTupleOp), "Expected tuple unpack operation after sql op array unpack"
            ncolumns.append(columns[target.tuple_idx])
            for edge in self.graph.edge_source[target]:
                self.graph.addEdge(SQLResultEdge(o, edge.target, edge, pos))

        o.setColumns(ncolumns)
        o.setFromObj(node.query)
        #}}}
    
    def eatOp(self, node):#{{{
        pass#}}}

    def eatDataOp(self, node):#{{{
        o = SQLValue(node.data, node)
        self.graph.addNode(o)
        self.next_round.add(o)
        self.sql_obj.add(o)

        for edge in self.graph.edge_source[node]:
            self.graph.addEdge(SQLResultEdge(o, edge.target, edge, 0))#}}}
    
    #SQL QUERY
    def eatSQLElement(self, node):#{{{
        self.expressionEat(node)#}}}

    #SQL VALUE
    def eatSQLValue(self, node):#{{{
        targets = [edge.target for edge in self.graph.edge_source[node]]
        if(len(targets) == 1 and isinstance(targets[0], ops.ConvertOp)):
            target = targets[0]
            edge = self.graph.getEdge(node, target)
            self.graph.dropEdge(edge)

            etargets = list(self.graph.edge_source[target])
            for edge in etargets:
                self.graph.addEdge(SQLResultEdge(node, edge.target, edge, 0))
            
            self.next_round.add(node)
        else:
            return self.expressionEat(node)#}}}

    #EXPRESSIONS
    def expressionEatSQLElement(self, node):#{{{
        targets = [edge.target for edge in self.graph.edge_source[node]]
        links = self.graph.node_attributes['links']
        expressions = set([links[target] for target in targets if target in links])

        for expression in expressions:
            if expression in self.done:
                continue
            if all([self.allSQLParams(in_slice, expression) for in_slice in expression.in_slices]):
                self.expressionEat(expression)#}}}

    def expressionEatExpression(self, node):#{{{
        pass#}}}
    def expressionEatMatchExpression(self, expression):
        ldimpathset, rdimpathset = expression.getDims(self.graph)
        if(len(ldimpathset) > 1 or len(rdimpathset) > 1):
            return
        if(len(ldimpathset.pop()) > 1 or len(rdimpathset.pop()) > 1):
            return

        cledge,credge = expression.getComparisonEdges(self.graph)
        ledges,redges,olnodes,ornodes = expression.getInfo(self.graph)

        scledge = self.getSQLEdge(cledge)
        scredge = self.getSQLEdge(credge)
        sledges = [self.getSQLEdge(ledge) for ledge in ledges]
        sredges = [self.getSQLEdge(redge) for redge in redges]

        lsources = list(set([sledge.source for sledge in sledges]))
        rsources = list(set([sredge.source for sredge in sredges]))
        clsource = scledge.source
        crsource = scredge.source

        if(len(lsources) > 1 or len(rsources) > 1):
            return
        lsource = lsources[0]
        rsource = rsources[0]
        if(not self.compatible(lsource,clsource) or not self.compatible(rsource,crsource)):
            return
        if(not lsource.conn == rsource.conn):
            return


        jointype = expression.getType()
        if not jointype in set(["inner","left","right"]):
            return 


        lcol = clsource.getColumn(scledge.pos)
        rcol = crsource.getColumn(scredge.pos)
        nqueryobj = lsources[0].equijoin(rsources[0], lcol, rcol)

        nlcolumns = len(lsource.columns)

        self.graph.addNode(nqueryobj)

        for onode, ledge in zip(olnodes,sledges):
            for edge in self.graph.edge_source[onode]:
                self.graph.addEdge(SQLResultEdge(nqueryobj,edge.target,edge, ledge.pos))
        
        for onode, redge in zip(ornodes,sredges):
            for edge in self.graph.edge_source[onode]:
                self.graph.addEdge(SQLResultEdge(nqueryobj,edge.target,edge, nlcolumns + redge.pos))

        self.next_round.add(nqueryobj)
        self.sql_obj.add(nqueryobj)
        self.done.add(expression)#}}}

    def expressionEatFilterExpression(self, expression):#{{{
        #check filter dim
        dimpathset = expression.getDims()
        if(len(dimpathset) > 1 or len(dimpathset.pop()) != 0):
            return

        fedges,cedge, onodes = expression.getInfo(self.graph)
        
        if(not isinstance(cedge.source.type,(rtypes.TypeBool,rtypes.TypeSlice))):
            return

        scedge = self.getSQLEdge(cedge) 
        sfedges = [self.getSQLEdge(fedge) for fedge in fedges]

        fsources = list(set([sfedge.source for sfedge in sfedges]))
        if(len(fsources) != 1 or not isinstance(fsources[0],SQLQuery)):
            return 

        fsource = fsources[0]
        csource = scedge.source

        nsource = fsource.copy()
        #check filter type
        if(isinstance(cedge.source.type,rtypes.TypeBool)):
            if(not isinstance(csource, SQLTerm)):
                return
            #check sql compatibility
            if not self.compatible(fsource,csource):
                return
            
            nsource = nsource.addWhere(csource)

        else: #typeslice
            if(not isinstance(csource, SQLValue)):
                return
            if not isinstance(csource.value,slice):
                return
            if not csource.value.step is None: 
                return
            start = csource.value.start
            stop = csource.value.stop
            if(not start is None and start < 0):
                return
            if(not stop is None and stop < 0):
                return
            
            nsource = nsource.addLimit(start, stop)
        

        self.graph.addNode(nsource)
        
        for onode, fedge in zip(onodes,sfedges):
            for edge in self.graph.edge_source[onode]:
                self.graph.addEdge(SQLResultEdge(nsource,edge.target,edge, fedge.pos))
        
        self.next_round.add(nsource)
        self.sql_obj.add(nsource)
        self.done.add(expression)#}}}
    
    def expressionEatBinFuncElemExpression(self, expression):#{{{
        sig, op = expression.getOp()
        #check sig
        #check sql compatibility
        if(not op in SQLOperators and not (sig + op) in SQLOperators):
            return
        if(op in SQLOperators):
            opfunc = SQLOperators[op]
        else:
            opfunc = SQLOperators[sig + op]

        #check dimensions
        leftedge,rightedge = expression.getLeftRightInEdges(self.graph)
        ldims, rdims = leftedge.source.dims, rightedge.source.dims
        if len(ldims) > 1 or len(rdims) > 1:
            return
        if ldims != rdims and len(rdims) == len(ldims):
            return
        
        sqlleftedge = self.getSQLEdge(leftedge)
        sqlrightedge = self.getSQLEdge(rightedge)

        left = sqlleftedge.source
        right = sqlrightedge.source

        if(not self.compatible(left, right)):
            return
       
        out = expression.getOutSlice()
        res = SQLBinTerm(left.getColumn(sqlleftedge.pos), right.getColumn(sqlrightedge.pos), opfunc, out)
        self.graph.addNode(res)
        self.graph.addEdge(SQLTransEdge(left, res, sqlleftedge, sqlleftedge.pos, 0))
        self.graph.addEdge(SQLTransEdge(right, res, sqlrightedge, sqlrightedge.pos, 1))

        out_edges = expression.getOutEdges(self.graph)
        for out_edge in out_edges:
            self.graph.addEdge(SQLResultEdge(res,out_edge.target,out_edge,0))

        self.next_round.add(res)
        self.sql_obj.add(res)
        self.done.add(expression)#}}}


    def finish(self, node):#{{{
        #filter out unnecessary fields
        used_pos = set()
        posresult = dict()
        results = []
        for edge in list(self.graph.edge_source[node]):
            if(isinstance(edge.realedge,query_graph.ParamEdge)):
                used_pos.add(edge.pos)
                posresult[edge.pos]= edge.realedge.source

        used_pos = list(used_pos)
        opos_to_npos = dict([(opos,npos) for npos, opos in enumerate(used_pos)])
        
        columns = [node.columns[pos] for pos in used_pos]
        results = [posresult[pos] for pos in used_pos]
        fnode = node.copy()
        fnode.setColumns(columns)

        #compile, create unpack slices
        tslice = ops.PackTupleOp(results)
        aslice = ops.PackArrayOp(tslice)
        
        sslice = SQLOp(fnode.conn, fnode.compile(), aslice.type)
        rslice = ops.UnpackArrayOp(sslice)
        self.addPairToGraph(sslice,rslice)
        
        rslices = []
        for oslice, i in zip(results, range(len(results))):
            r = ops.UnpackTupleOp(rslice,i)
            self.addPairToGraph(rslice,r)

            if(r.bookmarks):
                r2 = ops.ChangeBookmarkOp(r)
                r2.bookmarks = oslice.bookmarks
                self.addPairToGraph(r,r2)
                r= r2

            rslices.append(r)
      
        for edge in list(self.graph.edge_source[node]):
            self.graph.dropEdge(edge.realedge)
            self.graph.dropEdge(edge)
            
            redge = edge.realedge
            if(isinstance(redge,query_graph.ParamEdge)):
                redge.source = rslices[opos_to_npos[edge.pos]]
                self.graph.addEdge(redge)#}}}
        
        
SQLOperators = {'Equal':lambda x, y: x==y,
                'NotEqual': lambda x, y: x!=y,
                'Less': lambda x, y: x < y,
                'LessEqual': lambda x, y: x <= y,
                'Greater': lambda x,y: x > y,
                'GreaterEqual': lambda x,y: x >= y,
                'boolboolAnd': sql.and_,
                'boolboolOr': sql.or_,
                'boolboolXor': lambda x, y: x ^ y,
                'simple_arithAdd': lambda x, y: x + y,
                'string_add_stringAdd': lambda x, y: x + y,
                'Subtract': lambda x, y: x - y,
                'Multiply': lambda x, y: x * y,
                'Modulo': lambda x, y: x % y,
                "Divide": lambda x, y: x / y,
                'Power': lambda x, y: x ** y
                }


class SQLOp(ops.ExtendOp):#{{{
    __slots__ = []
    passes = [SQLPlanner]
    def __init__(self, conn, query, rtype, name="result"):
        self.conn = conn
        self.query = query
        ops.ExtendOp.__init__(self,name=name,rtype=rtype)

    def py_exec(self):
        if(isinstance(self.query, sqlalchemy.sql.expression.Executable)):
            res = self.conn.execute(self.query)
        else:
            res = self.conn.execute(sql.select([self.query]))
        result = res.fetchall()
        ndata = nested_array.NestedArray(result,self.type)
        return wrapper_py.ResultOp.from_slice(ndata,self)

    def __str__(self):
        if(isinstance(self.query, sqlalchemy.sql.expression.Executable)):
            return ops.ExtendOp.__str__(self) + ":\n" +  str(self.query.compile())
        else:
            return ops.ExtendOp.__str__(self) + ":\n" +  str(sql.select([self.query]).compile())#}}}


class SQLElement(ops.MultiOp):
    def getSource(self, graph):
        return self

class SQLQuery(SQLElement):#{{{
    def __init__(self, conn):
        self.conn = conn
        self.columns = []
        self.where = []
        self.limit = None
        self.offset = None
        self.from_obj = None
        
        SQLElement.__init__(self, tuple())

    def getColumn(self, pos):   
        return self.columns[pos]

    def setColumns(self, columns):
        self.columns = columns

    def setFromObj(self, from_obj):
        self.from_obj = from_obj

    def equijoin(self, other, lcol, rcol):
        assert self.conn == other.conn, "Different connections in join not allowed"
        s = SQLQuery(self.conn)
        other = other.to_subquery()

        s.columns = self.columns + other.columns
        s.where = self.where + other.where
        s.from_obj = newfromobj
        return s

    def compile(self):
        if(self.where):
            where=sql.and_(*self.where)
        else:
            where = None
        return sql.select(self.columns, whereclause=where,limit=self.limit,offset=self.offset, from_obj=self.from_obj,use_labels=True)

    def copy(self):
        return copy.copy(self)

    def to_subquery(self):
        ns = SQLQuery(self.conn)
        q = self.compile().alias()
        ns.columns = list(q.columns)
        ns.from_obj = q
        return ns

    def addWhere(self, where):
        assert isinstance(where, SQLTerm), "Adding non-term where expression"
        self.where.append(where.compile())
        return self


    def __str__(self):
        return str(self.compile())#}}}

class SQLValue(SQLElement):#{{{
    def __init__(self, value, slice):
        self.value = value
        SQLElement.__init__(self, (slice,))

    def getColumn(self, pos):
        assert pos == 0, "Illegal pos for SQLValue"
        return self.value

    def compile(self):
        return self.value

    def __str__(self):
        return str(self.value)#}}}

class SQLTerm(SQLElement):#{{{
    def getSource(self, graph):
        tedges = [edge for edge in graph.edge_target[self] if isinstance(edge, SQLTransEdge)]
        sources = set([edge.source.getSource(graph) for edge in tedges])

        fsources = [source for source in sources if not isinstance(source, SQLValue)]
        if not fsources:
            return None
        else:
            assert len(fsources) == 1, "Found multiple SQLQuery sources for SQLOp"
            return fsources[0]#}}}

class SQLBinTerm(SQLTerm):#{{{
    def __init__(self, left, right, op, slice):
        self.op = op
        self.left = left
        self.right = right
        SQLTerm.__init__(self, (slice,))

    def compile(self):
        return self.op(self.left, self.right)
    
    def getColumn(self, pos):
        assert pos == 0, "Illegal pos for SQLBinTerm"
        return self.compile()

    def __str__(self):
        return str(self.compile())#}}}


class Column(object):
    def __init__(self, table, name):
        self.table = table
        self.name = name

    def realias(self, realias_dict):
        if(self.table in realias_dict):
            return Column(realias_dict[self.table],self.name)
        else:
            return self

    def recol(self, column_dict):
        return column_dict.get(self,self)

    def compile(self):
        c = list(self.table.compile().columns)
        for col in cols:
            if col.name == self.name:
                return col
        raise RuntimeError, "Could not find colunm " + str(self.name) + " in table " + str(self.table.compile())


    def getTables(self):
        return set([self.table])

class Value(Column):
    def __init__(self, data):
        self.data = data

    def compile(self):
        return self

    def realias(self, realias_dict):
        return self

    def recol(self, realias_dict):
        return self

    def getTables(self):
        return set()

class Term(Column):
    def __init__(self, func, *sources):
        self.sources = sources
        self.func = func

    def realias(self, realias_dict):
        nsources = [source.realias(realias_dict) for source in self.sources]
        return Term(nsources, self.func)
        
    def recol(self, column_dict):
        nsources = [source.recol(column_dict) for source in self.sources]
        return Term(nsources, self.func)

    def compile(self):
        nc = [source.compile() for source in sources]
        return self.func(*nc)

    def getTables(self):
        return reduce(operator.__or__,[source.getTables() for source in self.sources],set())


class Table(object):
    def __init__(self, source_descriptor):
        self.source_descriptor = source_descriptor

    def getSource(self):
        return self.source_descriptor

    def getColumns(self):
        cols = list(self.getSource().columns)
        return [Column(self, colname) for col in cols]

    def realias(self, realias_dict):
        return realias_dict.get(self, self)

    def alias(self, realias_dict):
        n = AliasTable(self)
        realias_dict[self] = n
        return n
    
    def getTables(self):
        return set([self])

    def compile(self):
        return self.source_descriptor

class AliasTable(Table):
    def __init__(self, origtable):
        self.origtable = origtable
        Table.__init__(self,origtable.getSource().alias())

    def alias(self, realias_dict):
        return self.origtable.alias(realias_dict)
 
class Join(Table):
    def __init__(self, left, right, jointype, condition):
        assert not left.getTables() & right.getTables(), "Overlap in tables in join condition"
        assert not (self.condition.getTables() - (left.getTables() | right.getTables())), "Condition has tables outside join"
        self.left = left
        self.right = right
        self.jointype = jointype
        self.condition = condition

    def realias(self, realias_dict):
        n = Join(self.left.realias(realias_dict), self.right.realias(realias_dict), self.jointype, self.condition.realias(realias_dict))

    def getTables(self):
        return self.left.getTables() | self.right.getTables() | self.condition.getTables()

    def getColumns(self):
        return self.left.getColumns() + self.right.getColumns()

    def compile(self):
        lsource = self.left.compile()
        rsource = self.right.compile()
        if(jointype == "inner"):
            fromobj = lsource.from_obj.join(rsource.from_obj, onclause=self.condition)
        elif(jointype == "left"):
            fromobj = lsource.from_obj.join(rsource.from_obj, onclause=self.condition,isouter=True)
        elif(jointype == "right"):
            fromobj = rsource.from_obj.join(lsource.from_obj, onclause=self.condition,isouter=True)
        return fromobj


class Query(Table):
    def __init__(self, conn, from_obj):
        self.conn = conn
        self.from_obj = from_obj
        self.columns = self.from_obj.getColumns()
        self.limit = None
        self.offset = None
        self.distinct = False
        self.whereclause = []
        self.groupby_clause = []
        self.havingclause = []
        self.orderedby_clause = []

    def setColumns(self, cols):
        assert not (reduce(operator.__or__,[col.getTables() for col in cols],set()) - self.from_obj.getTables()), "Column tables not in from obj"
        self.columns = cols

    def _clause_subquery(self, clause):
        nself = self.alias()
        column_dict = dict(zip(self.columns,nself.columns))
        return (nself, clause.recol(column_dict))

    def addWhere(self, clause):
        if (self.groupby_clause or self.limit or self.offset):
            self, clause = self._clause_subquery(clause)

        assert not clause.getTables() - self.from_obj.getTables(), "Where tables not in from obj"
        self.whereclause.append(clause)
        return self

    def addHaving(self, clause):
        assert self.groupby_clause and not (self.limit or self.offset), "Having clause should be added to queries with group, and without offset/limit"
        assert not clause.getTables() - self.from_obj.getTables(), "Having tables not in from obj"
        self.havingclause.append(clause)
        return self

    def addGroupBy(self, clause):
        if (self.limit or self.offset):
            assert not self.groupby_clause, "Cannot perform group as limit or offset has been set inbetween"
            self, clause = self._clause_subquery(clause)

        assert not clause.getTables() - self.from_obj.getTables(), "Group tables not in from obj"
        self.groupby_clause.append(clause)
        return self

    def addOrderedBy(self, clause):
        if (self.limit or self.offset or self.groupby_clause):
            assert not self.orderedby_clause, "Cannot perform order as limit or offset has been set inbetween"
            self, clause = self._clause_subquery(clause)
 
        assert not clause.getTables() - self.from_obj.getTables(), "Group tables not in from obj"
        self.orderedby_clause.append(clause)
     
    def setDistinct(self, distinct):
        assert not (self.limit or self.offset or self.groupby_clause), "Cannot set distinct on query with group"
        self.distinct = distinct
        return self

    def addLimit(self, start, stop):
        assert start is None or start >= 0, "Start cannot be smaller than 0"
        assert stop is None or stop >= 0, "Stop cannot be smaller than 0"

        if(not start is None):
            xoffset = max(self.offset,0)#None to 0
            if(self.limit is None):
                self.offset = xoffset + start
            else:
                start = min(start, self.limit)
                self.offset = xoffset + start
                self.limit -= start
            
        if(not stop is None):
            if(self.limit is None):
                self.limit = stop - max(start,0)
            else:
                self.limit = min(stop - max(start,0), self.limit)
        return self


    def realias(self, realias_dict):
        n = Query(self.conn, self.from_obj.realias(realias_dict))
        n.columns = [col.realias(realias_dict) for col in self.columns]
        n.whereclause = [wc.realias(realias_dict) for wc in self.whereclause]
        n.groupby_clause = [gc.realias(realias_dict) for gc in self.groupby_clause]
        n.orderedby_clause = [oc.realias(realias_dict) for oc in self.orderedby_clause]
        n.havingclause = [hc.realias(realias_dict) for hc in self.havingclause]
        n.limit = self.limit
        n.offset = self.offset
        n.distinct = self.distinct
        return n

    def getTable(self):
        return self

    def getTables(self):
        return self.from_obj.getTables()

    def alias(self, realias_dict):
        n = AliasQuery(self.conn, self)
        realias_dict[self] = n
        n.setColumns([Column(x, col.name) for col in self.columns])
        return n

    def _from_compile(self):
        return self.from_obj.compile()

    def compile(self):
        columns = [c.compile() for c in self.columns]
        from_obj = self._from_compile()
        whereclause = sql.and_(*[wc.compile() for wc in self.whereclause])
        return sql.select([columns],whereclause=whereclause, from_obj = from_obj, limit=self.limit, offset = self.offset, distinct=self.distinct, use_labels = True)

class AliasQuery(Query, Table):

    def getColumns(self):
        return self.columns

    def realias(self, realias_dict):
        self = realias_dict[self]
        self = self.from_obj.realias(realias_dict)
        return self

    def alias(self, realias_dict):
        return self.from_obj.alias(realias_dict)

    def getSource(self):
        return self.from_obj

    def getTables(self):
        return set([self]) | self.from_obj.getTables()

    def _from_compile(self):
        return self.from_obj.compile().alias()

