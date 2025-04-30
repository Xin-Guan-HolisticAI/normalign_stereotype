from normalign_stereotype.core._objects._reference import Reference
from typing import Optional

# Concept type constants
CONCEPT_TYPE_CLASSIFICATION = "?"
CONCEPT_TYPE_JUDGEMENT = "<>"
CONCEPT_TYPE_RELATION = "[]"
CONCEPT_TYPE_OBJECT = "{}"
CONCEPT_TYPE_SENTENCE = "^"
CONCEPT_TYPE_ASSIGNMENT = "@"

CONCEPT_TYPES = {
    CONCEPT_TYPE_CLASSIFICATION: "classification",
    CONCEPT_TYPE_JUDGEMENT: "judgement",
    CONCEPT_TYPE_RELATION: "relation",
    CONCEPT_TYPE_OBJECT: "object",
    CONCEPT_TYPE_SENTENCE: "sentence",
    CONCEPT_TYPE_ASSIGNMENT: "assignment"
}

class Concept:
    def __init__(self, name, context="", reference=None, type=CONCEPT_TYPE_OBJECT):
        if type is not None and type not in CONCEPT_TYPES:
            raise ValueError(f"Invalid concept type. Must be one of: {list(CONCEPT_TYPES.keys())}")
            
        # Comprehension attribute (required)
        self.comprehension = {
            "name": name,
            "context": context,
            "type": type,
            "type_description": CONCEPT_TYPES.get(type, None)
        }

        # Reference attribute (optional)
        self.reference: Reference = reference

    def read_reference_from_file(self, path):
        # Load reference tensor from file
        concept_name = self.comprehension["name"]

        ref_tensor = eval(open(path, encoding="utf-8").read())

        # Store reference tensor in global namespace
        globals()[f"{concept_name}_ref_tensor"] = ref_tensor

        # Create and configure Reference object
        reference = Reference(
            axes=[concept_name],
            shape=(len(ref_tensor),),
            initial_value=0
        )
        reference.tensor = ref_tensor
        globals()[f"{concept_name}_ref"] = reference

        self.reference = reference


