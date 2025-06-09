import subprocess
import sys

def test_odgr_executor_expertbasedgraml_minigrid():
    """Test odgr_executor.py with ExpertBasedGraml on minigrid, L1, easiest env."""
    result = subprocess.run([
        sys.executable, "gr_libs/odgr_executor.py",
        "--domain", "minigrid",
        "--env_name", "MiniGrid-SimpleCrossingS13N4",
        "--recognizer", "ExpertBasedGraml",
        "--task", "L1",
        "--collect_stats"
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"ExpertBasedGraml minigrid L1 failed: {result.stderr}"
