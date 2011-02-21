"""Engines used to process data."""
import passes
from wrappers import wrapper_py

select_engine = passes.PassManager(log=False)
select_engine.register(passes.EnsureInfo)
select_engine.register(passes.CreateGraph)
select_engine.register(passes.AnnotateRepLinks)
select_engine.register(passes.PrePeepHole)
select_engine.register(passes.CreateExpressions)
select_engine.register(passes.WrapperPlanner,debug_mode=False)
select_engine.register(passes.SerializeExec)
select_engine.register(wrapper_py.PyExec,debug_mode=False)
#select_engine.register(passes.DebugVisualizer)

