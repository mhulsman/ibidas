"""Engines used to process data."""
import passes
from wrappers import python

select_engine = passes.PassManager(log=False)
select_engine.register(passes.EnsureInfo)
select_engine.register(passes.CreateGraph)
select_engine.register(passes.AnnotateRepLinks)
#select_engine.register(passes.DebugVisualizer)
select_engine.register(passes.PrePeepHole)
select_engine.register(passes.CreateExpressions)
select_engine.register(passes.WrapperPlanner,debug_mode=False)
select_engine.register(passes.PythonPeepHole)
#select_engine.register(passes.DebugVisualizer)
select_engine.register(passes.Transposer)
#select_engine.register(passes.DebugVisualizer)
select_engine.register(passes.SerializeExec)
select_engine.register(python.PyExec,debug_mode=False)
#select_engine.register(passes.DebugVisualizer)

