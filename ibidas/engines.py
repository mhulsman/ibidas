"""Engines used to process data."""
import passes

select_engine = passes.PassManager()
select_engine.register(passes.TargetCalc)
select_engine.register(passes.PreOrderWalk)
#select_engine.register(passes.DebugVisualizer)
select_engine.register(passes.RequiredSliceIds)
select_engine.register(passes.PlanWrapper)

