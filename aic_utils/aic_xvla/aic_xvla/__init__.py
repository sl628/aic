"""aic_xvla package.

Submodules are intentionally NOT imported here so that submodules with
heavy / env-specific dependencies (handler, eval, serve — all need X-VLA)
do not break import in environments that only run the ROS policy
(aic_xvla.ros.RunXVLA, which only needs requests + cv2).

Import what you need explicitly:

    from aic_xvla.handler import AICHandler, register   # X-VLA env only
    from aic_xvla.eval import AICXVLAPolicy              # X-VLA env only
    from aic_xvla.ros.RunXVLA import RunXVLA             # aic pixi env

The aic_model node loads policies via:
    importlib.import_module("aic_xvla.ros.RunXVLA")
then picks the class whose name matches the last segment ("RunXVLA").
So the ROS parameter is `policy:=aic_xvla.ros.RunXVLA` (NOT
`aic_xvla.ros.RunXVLA.RunXVLA`).
"""
