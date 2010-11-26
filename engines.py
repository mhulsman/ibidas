"""Engines used to process data."""
import pass_manager
import pass_target
import pass_prewalk
import pass_required
import pass_debug
import pass_planwrapper

select_engine = pass_manager.PassManager()
select_engine.register(pass_target.TargetCalc)
select_engine.register(pass_prewalk.PreOrderWalk)
#select_engine.register(pass_debug.DebugVisualizer)
select_engine.register(pass_required.RequiredSliceIds)
select_engine.register(pass_planwrapper.PlanWrapper)

