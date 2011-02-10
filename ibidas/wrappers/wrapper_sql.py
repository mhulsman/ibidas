import wrapper
import sqlalchemy
from sqlalchemy.sql import expression as sql

from .. import ops
from ..constants import *
from ..utils.multi_visitor import VisitorFactory, NF_ELSE, F_CACHE, NF_ERROR

from ..passes import manager, create_graph, annotate_replinks, serialize_exec
from .. import query_graph

_delay_import_(globals(),"..utils","util","nested_array")
_delay_import_(globals(),"..itypes","rtypes","dimensions","dimpaths")
_delay_import_(globals(),"wrapper_py")

class TypeMapperToSQLAlchemy(VisitorFactory(prefixes=("convert",), 
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
            return sqlalchemy.types.Text()
tosa_typemapper = TypeMapperToSQLAlchemy()


class TypeMapperFromSQLAlchemy(VisitorFactory(prefixes=("convert",), 
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
        return rtypes.createType(typestr)

sa_typemapper = TypeMapperFromSQLAlchemy()

postgres_types = {}
def _getPostgresTypes(engine):
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
    return postgres_types[engine]
pgtoibidas = {'int2':'int16','varchar':'string[]','int4':'int32','text':'string[]', 'float4':'real32', 'float8':'real64'}


mysql_types = {}
def _getMySQLTypes(engine):
    """Returns a list of dictionaries with 
        information on each type in the database"""
    return mysqlid_toibidas
mysqlid_toibidas = {0:'real64',1:'int8',2:'int16',3:'int32',4:'real32',5:'real64',6:'any',7:'any',8:'int64',9:'int32',10:'any',
                    11:'any', 12:'any',13:'uint16',14:'any',15:'string[]',16:'bool',246:'any',247:'any',248:'any',249:'any',250:'any',
                    251:'any',252:'any',253:'string[]',254:'string[]',255:'any'};

def convert(col_descriptor, type, engine, tablename=None):
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
    return table_type
        

def convert_mysql(col_descriptor, engine):
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
    return (fieldnames, subtypes)
   

def convert_postgres(col_descriptor, engine):
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
    return (fieldnames, subtypes)

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
    def __init__(self, engine, schemaname=None):
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
                pass

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


    def store(self, name, rep, primary_slices=None, indexes = {}, unique = {}):
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
        return self.tabledict[name]
    
    def _append(self, name, rep):
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
       self.engine.execute(self.sa_tables[name].insert(), values)

    def drop(self, name):
        self.sa_tables[name].drop(bind=self.engine, checkfirst=True)
        del self.sa_tables[name]
        del self.tabledict[name]

class TableRepresentor(wrapper.SourceRepresentor):
    def __init__(self, engine, table):
        table_type = convert(table.columns,'sqlalchemy',engine, table.name)
        nslices = (SQLOp(engine, table, table_type, table.name),)
        self._initialize(nslices,RS_ALL_KNOWN)

class QueryRepresentor(wrapper.SourceRepresentor):
    def __init__(self, engine, query):
        res = self.engine.execute(query)
        try:
            query_type = convert(res.context.compiled.statement.columns, "sqlalchemy", self.engine)
        except AttributeError:
            query_type = convert(res.cursor.description, res.dialect.name, self.engine)
        res.close() 
        
        nslices = (SQLOp(engine, query, query_type),)
        self._initialize(nslices,RS_ALL_KNOWN)



class SQLResultEdge(query_graph.Edge):
    __slots__ = ["realedge","pos"]
    def __init__(self, source, target, realedge, fieldpos):
        self.realedge = realedge
        self.pos = fieldpos
        query_graph.Edge.__init__(self, source, target)

class SQLPlanner(VisitorFactory(prefixes=("eat","finish"), 
                                      flags=NF_ERROR | F_CACHE), manager.Pass):
    after=set([create_graph.CreateGraph, annotate_replinks.AnnotateRepLinks])
    before=set([serialize_exec.SerializeExec])

    @classmethod
    def run(cls, query, run_manager):
        self = cls()
        self.graph = run_manager.pass_results[create_graph.CreateGraph]
        self.expressions = run_manager.pass_results[annotate_replinks.AnnotateRepLinks]

        self.eat_front = self.graph.getSources()

        changes = True
        while(changes and self.eat_front):
            changes=False
            for node in list(self.eat_front):
                changes = changes or self.eat(node) 

        for node in self.eat_front:
            self.finish(node)


    def eatSQLOp(self, node):
        o = SQLQuery(node.conn)
        self.graph.addNode(o)

        self.eat_front.discard(node)
        self.eat_front.add(o)

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
        o.setResults(tuple(targets))
        return True
    
    def eatOp(self, node):
        self.eat_front.discard(node)
        return True

    def eatSQLQuery(self, node):
        return False


    def finishSQLQuery(self, node):
        query = node.compile()
        tslice = ops.PackTupleOp(node.results)
        aslice = ops.PackArrayOp(tslice)
        
        sslice = SQLOp(node.conn, node.compile(), aslice.type)
        rslice = ops.UnpackArrayOp(sslice)
        self.addPairToGraph(sslice,rslice)
        
        rslices = []
        for oslice, i in zip(node.results, range(len(node.results))):
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
            redge.source = rslices[edge.pos]
            self.graph.addEdge(redge)
        
    def addPairToGraph(self, node, node2):
        self.graph.addNode(node)
        self.graph.addNode(node2)
        self.graph.addEdge(query_graph.ParamEdge(node,node2,"slice"))
        


class SQLOp(ops.ExtendOp):
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
            return ops.ExtendOp.__str__(self) + ":\n" +  str(sql.select(self.query).compile())



class SQLQuery(ops.MultiOp):
    def __init__(self, conn):
        self.conn = conn
        self.columns = []
        
        ops.MultiOp.__init__(self, tuple())


    def setColumns(self, columns):
        self.columns = columns

    def setResults(self, slices):
        self.results = slices

    def compile(self):
        return sql.select(self.columns)






