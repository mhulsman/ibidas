"""Engines used to process data."""
import passes
from wrappers import wrapper_py

select_engine = passes.PassManager()
select_engine.register(passes.EnsureInfo)
select_engine.register(passes.CreateGraph)
select_engine.register(passes.SerializeExec)
select_engine.register(wrapper_py.PyExec,debug_mode=True)
#select_engine.register(passes.DebugVisualizer)
#select_engine.register(passes.TargetCalc)
#select_engine.register(passes.PreOrderWalk)
#select_engine.register(passes.RequiredSliceIds)
#select_engine.register(passes.PlanWrapper)

