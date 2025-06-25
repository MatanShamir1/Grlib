import subprocess
import sys

def test_odgr_executor_gcaura_parking():
    """Test odgr_executor.py with GCAura on parking, L1, easiest env."""
    result = subprocess.run([
        sys.executable, "gr_libs/odgr_executor.py",
        "--domain", "parking",
        "--env_name", "Parking-S-14-PC-",
        "--recognizer", "GCAura",
        "--task", "L1",
        "--collect_stats"
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"GCAura parking L1 failed: {result.stderr}"
