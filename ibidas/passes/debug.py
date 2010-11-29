import manager
from ..utils.multi_visitor import VisitorFactory, F_CACHE, NF_ROBJ

_delay_import_(globals(),"..repops")

class DebugVisualizer(VisitorFactory(prefixes=("visit", "target"),
                                     flags=F_CACHE | NF_ROBJ), 
                      manager.Pass):
    @classmethod
    def run(cls, query, pass_results):
        #delayed import, only import if debug is enabled
        import pygraphviz as pg
        from ibidas.thirdparty import xdot
        import gtk
        import gtk.gdk

        self = cls()
        self.graph = pg.AGraph(strict=False, directed=True);

        self.visit(query.root)
        #self.target(query.qg)
        try:
            window = xdot.DotWindow()
            window.set_dotcode(self.graph.draw(format="xdot", prog="dot"))
            window.connect('destroy', gtk.main_quit)
            gtk.main()
        except IOError, e:
            #when using xlib, on closing of the screen an empty ioerror is send.
            #this error is filtered out here
            if(len(str(e)) > 0):
                raise e

    def visitRepresentor(self, crep):
        lbl = "{" + crep.__class__.__name__

        lbl += "|{" + "|".join([slice.name + ":" + str(slice.id) + ":" + \
                                str(slice.last_id) 
                                for slice in crep._active_slices]) + "}"
        lbl += "|{" + "|".join([str(slice.type) 
                                for slice in crep._active_slices]) + "}"
        lbl += "|{" + "|".join([str(slice.dims) 
                                for slice in crep._active_slices]) + "}"
        lbl += "|{" + str(crep._all_slices) + "}"
       
        lbl += "}"
        lbl = lbl.replace(" ","\ ")
        lbl = lbl.replace(">","\>")
        lbl = lbl.replace("\n","\\n")

        self.graph.add_node(id(crep), fontsize="8", shape="record", 
                            label=lbl, rankdir="LR")
        if(isinstance(crep, repops.OpRep)):
            for source in crep._sources:   
                s_id = self.visit(source)
                self.graph.add_edge(s_id, id(crep))
        
        return id(crep)
    
    def targetnode(self, node):
        self.targetsources(node)
        for target in node.target:
            self.graph.add_edge(id(target), id(node), 
                                color="red", constraint="false")
            

