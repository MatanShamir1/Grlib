from gr_libs.tutorials.Draco.draco_panda_tutorial import run_draco_panda_tutorial
from gr_libs.tutorials.Draco.draco_parking_tutorial import run_draco_parking_tutorial


def test_draco_panda_tutorial():
    run_draco_panda_tutorial()


def test_draco_parking_tutorial():
    run_draco_parking_tutorial()

if __name__ == "__main__":
    test_draco_panda_tutorial()
    test_draco_parking_tutorial()