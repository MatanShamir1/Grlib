import subprocess
import sys

def test_odgr_executor_gcgraml_parking():
    """Test odgr_executor.py with GCGraml on parking, L1, easiest env."""
    result = subprocess.run([
        sys.executable, "gr_libs/odgr_executor.py",
        "--domain", "parking",
        "--env_name", "Parking-S-14-PC-",
        "--recognizer", "GCGraml",
        "--task", "L1",
        "--collect_stats"
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"GCGraml parking L1 failed: {result.stderr}"
