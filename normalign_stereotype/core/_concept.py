from normalign_stereotype.core._reference import Reference
from typing import Optional

def create_concept_reference(concept: str, value: str, summary: Optional[str] = None) -> Reference:
    """Create a reference for a concept with an explicit value.
    
    Args:
        concept: The name of the concept
        value: The explicit value to assign to the concept
        summary: Optional summary for the reference. If None, uses value
        
    Returns:
        A Reference object containing the concept reference with the specified value
    """
    if summary is None:
        summary = value

    return Reference(
            axes=[concept],
            shape=(1,),
            initial_value=f"[{value} :{summary}]"
        )


class Concept:
    def __init__(self, name, context="", reference=None):
        # Comprehension attribute (required)
        self.comprehension = {
            "name": name,
            "context": context
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


