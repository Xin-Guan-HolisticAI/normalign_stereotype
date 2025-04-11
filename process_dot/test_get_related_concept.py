import os
from plan_with_dot import DOTParser

def test_get_related_concepts():
    # Initialize the parser with the metaphor_draft.dot file
    dot_file = os.path.join(os.path.dirname(__file__), "metaphor_draft.dot")
    parser = DOTParser(dot_file)
    
    # Parse the DOT file
    parser.parse()
    
    # Test cases - you can add more concepts to test
    test_concepts = [
        "metaphor",  # A central concept
        "source_domain",  # A base concept
        "target_domain",  # Another base concept
        "mapping",  # A concept that connects domains
    ]
    
    # Test each concept
    for concept in test_concepts:
        print(f"\nTesting concept: {concept}")
        try:
            related = parser.get_related_concepts(concept)
            
            print(f"Incoming perception concepts: {related['incoming_perception']}")
            print(f"Incoming actuation concepts: {related['incoming_actuation']}")
            print(f"Outgoing perception concepts: {related['outgoing_perception']}")
            print(f"Outgoing actuation concepts: {related['outgoing_actuation']}")
            
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_get_related_concepts() 