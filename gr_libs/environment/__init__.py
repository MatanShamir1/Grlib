import importlib.metadata
import warnings

def is_extra_installed(package: str, extra: str) -> bool:
    """Check if an extra was installed for a given package."""
    try:
        # Get metadata for the installed package
        dist = importlib.metadata.metadata(package)
        requires = dist.get_all("Requires-Dist", [])  # Dependencies listed in the package metadata
        return any(extra in req for req in requires)
    except importlib.metadata.PackageNotFoundError:
        return False  # The package is not installed

# Check if `gr_libs[minigrid]` was installed
for env in ["minigrid", "panda", "highway", "point_maze"]:
	if is_extra_installed("gr_libs", f"gr_envs[{env}]"):
		try:
			importlib.import_module(f"gr_envs.{env}_scripts.envs")
		except ImportError:
			raise ImportError(f"gr_envs[{env}] was not installed, but gr_libs[{env}] requires it! if you messed with gr_envs installation, you can reinstall gr_libs.")
	else:
		warnings.warn(f"gr_libs[{env}] was not installed, skipping {env} imports.", RuntimeWarning)
