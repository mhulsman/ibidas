import SimpleXMLRPCServer
from utils import util
from representor import Representor
import thread

def startServer(portnumber=9889):
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
            res = res.to_python()
        
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

