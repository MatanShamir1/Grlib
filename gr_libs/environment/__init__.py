"""
A module GR algorithms can store hard-coded parameters and functionalities
that are environment-related. We try to import to this level every environment
that is an extra, so that users that installed `gr_libs` with the extra
`[minigrid]`, `[panda]`, `[highway]` or `[maze]` can use it directly
from `gr_libs.environment` without having to import `gr_envs`.
If the extra was not installed, a warning is raised.
"""

import importlib
import warnings

# Track which extras/environments are available
AVAILABLE_DOMS = {}

for extra, dom in zip(
    ["minigrid", "panda", "highway", "maze"],
    ["minigrid", "panda", "parking", "point_maze"],
):
    try:
        importlib.import_module(f"gr_envs.{extra}_scripts.envs")
        AVAILABLE_DOMS[dom] = True
    except ImportError:
        AVAILABLE_DOMS[dom] = False
        warnings.warn(
            f"gr_envs[{extra}] is not installed. If you need it, install gr_libs with the [{extra}] extra.",
            RuntimeWarning,
        )
