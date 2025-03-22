from gr_libs.recognizer.graml.graml_recognizer import GCGraml, ExpertBasedGraml, MCTSBasedGraml
from gr_libs.recognizer.gr_as_rl.gr_as_rl_recognizer import Graql, Draco, GCDraco

def recognizer_str_to_obj(recognizer_str: str):
    recognizer_map = {
        "GCGraml": GCGraml,
        "ExpertBasedGraml": ExpertBasedGraml,
        "MCTSBasedGraml": MCTSBasedGraml,
        "Graql": Graql,
        "Draco": Draco,
        "GCDraco": GCDraco
    }
    return recognizer_map.get(recognizer_str)