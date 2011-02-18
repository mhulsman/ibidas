import SimpleXMLRPCServer
from utils import util
from representor import Representor
import thread

def Serve(portnumber=9889):
    """Starts xml-rpc server on the supplied port number. Can be used to contact ibidas
       from other applications.
       
       XML-RPC interface:

       * startSession() (optional): create session namespace, returns sessionid

       * closeSession(sessionid): closes specified session

       * query(query, sessionid=-1): executes query string in session, returns result.
       
       * execute(query, sessionid=-1): exexutes query string in session

       * set(name, val, sessionid=-1): sets variable in session
        
       The default session id of -1 gives direct contact with the ibidas command line session

       Starting the server::
           
           $ ibidas
           >>> s = Serve()
           >>> z = Rep([1,2,3,4])
      
       Connecting to ibidas (python, but can be any other application supporting xml-rpc)::
           $ipython
           >>> import xmlrpclib
           >>> s = xmlrpclib.ServerProxy("http://localhost:9889")
           >>> s.query("z + 4")
           [4,5,6,7]

       """

    s = SimpleXMLRPCServer.SimpleXMLRPCServer(("localhost",portnumber),allow_none=True,bind_and_activate=False, logRequests=False)
    s.allow_reuse_adress = True
    s.server_bind()
    s.server_activate()
    l = LocalRequestHandler()
    s.register_instance(l,allow_dotted_names=False)
    s.register_introspection_functions()
    s.register_multicall_functions()
    thread.start_new_thread(runServer,(s,))
    return s

def runServer(s):
    s.serve_forever()


class LocalRequestHandler(object):
    def __init__(self):
        self.sessions = []

    def startSession(self):
        if None in self.sessions:
            sessionid = self.sessions.index(None)
        else:
            sessionid = len(self.sessions)
            self.sessions.append(None)
        self.sessions[sessionid] = {}
        return sessionid

    def closeSession(self, sessionid):
        assert isinstance(sessionid,int), "Session id should be an int"
        assert sessionid >= 0 and sesionid < len(self.sessions), "Session id not valid"
        self.sessions[sessionid] = None

    def query(self, query, sessionid=-1):
        assert isinstance(sessionid,int), "Session id should be an int"
        assert sessionid >= -1 and sessionid < len(self.sessions), "Session id not valid"
      
        session = getSession(self, sessionid)

        res = eval(query,getGlobals(), session)
        if(isinstance(res,Representor)):
            res = res.ToPython()
        
        return res

    def execute(self, statement, sessionid=-1):
        assert isinstance(sessionid,int), "Session id should be an int"
        assert sessionid >= -1 and sessionid < len(self.sessions), "Session id not valid"

        session = getSession(self, sessionid)
       
        exec statement in getGlobals(), session

        return True

    def set(self, name, value, sessionid=-1):
        session = getSession(self, sessionid)
        session[name] = value


def getGlobals():
    if '__IPYTHON__active' in globals()['__builtins__']:
        return globals()['__builtins__']['__IPYTHON__'].user_global_ns
    else:
        return globals()

def getSession(rh, sessionid):
    if(sessionid == -1):
        assert '__IPYTHON__active' in globals()['__builtins__'], "Ipython not active, cannot find shell session"
        session = globals()['__builtins__']['__IPYTHON__'].user_ns
    else:
        session = rh.sessions[sessionid]
    return session

