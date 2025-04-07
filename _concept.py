from _reference import Reference

class Concept:
    def __init__(self, name, context="", reference=None):
        # Comprehension attribute (required)
        self.comprehension = {
            "name": name,
            "context": context
        }

        # Reference attribute (optional)
        self.reference = reference

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


