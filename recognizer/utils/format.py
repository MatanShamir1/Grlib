from recognizer.graml.graml_recognizer import GramlRecognizer
from recognizer.graql.graql_recognizer import GraqlRecognizer

def recognizer_str_to_obj(recognizer_str: str):
    if recognizer_str == "graml":
        return GramlRecognizer
    elif recognizer_str == "graql":
        return GraqlRecognizer
    elif recognizer_str == "draco":
        return GraqlRecognizer