import subprocess
import sys

def test_odgr_executor_gcdraco_parking():
    """Test odgr_executor.py with GCDraco on parking, L1, easiest env."""
    result = subprocess.run([
        sys.executable, "gr_libs/odgr_executor.py",
        "--domain", "parking",
        "--env_name", "Parking-S-14-PC-",
        "--recognizer", "GCDraco",
        "--task", "L1",
        "--collect_stats"
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"GCDraco parking L1 failed: {result.stderr}"
