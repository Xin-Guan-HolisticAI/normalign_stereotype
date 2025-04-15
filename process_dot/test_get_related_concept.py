import os
from plan_with_dot import DOTParser

def test_get_related_concepts():
    # Initialize the parser with the metaphor_draft.dot file
    dot_file = os.path.join(os.path.dirname(__file__), "metaphor_draft.dot")
    parser = DOTParser(dot_file)
    
    # Parse the DOT file
    parser.parse()

    print("\n=== Base Concepts ===")
    print(parser.base_concept_names)
    
    # Test cases for different types of concepts
    test_concepts = [
        # Base concepts (no incoming perception)
        "extract",
        "if",
        "that_$1_present",
        "that_$1_transfers_symbolic_meaning_to_$2_without_intermediaries",
        
        # Regular concepts with both perception and actuation
        "figurative_language_element",
        "specific_tangible_entity",
        "abstract_complex_theme",
        
        # Classification concepts
        "figurative_language_element?",
        "specific_tangible_entity?",
        "abstract_complex_theme?",
        
        # Complex concepts with multiple dependencies
        "that_figurative_language_element_maps_specific_tangible_entity_onto_abstract_complex_theme_directly"
    ]
    
    # Test each concept
    for concept in test_concepts:
        print(f"\n=== Testing concept: {concept} ===")
        try:
            related = parser.get_related_concepts(concept)
            
            print(f"Incoming perception concepts: {related['incoming_perception']}")
            print(f"Incoming actuation concepts: {related['incoming_actuation']}")
            print(f"Outgoing perception concepts: {related['outgoing_perception']}")
            print(f"Outgoing actuation concepts: {related['outgoing_actuation']}")
            
            # Additional validation
            if concept in parser.base_concept_names:
                assert not related['incoming_perception'], f"Base concept {concept} should not have incoming perception"
            
            if "?" in concept:
                assert related['outgoing_actuation'], f"Classification concept {concept} should have outgoing actuation"
            
        except ValueError as e:
            print(f"Error: {e}")

def test_actuation_context():
    """Test the actuation context functionality."""
    dot_file = os.path.join(os.path.dirname(__file__), "metaphor_draft.dot")
    parser = DOTParser(dot_file)
    parser.parse()
    
    # Get all actuation edges
    actuation_edges = [edge for edge in parser.edges if edge[2] == 'actu']
    
    print("\n=== Testing Actuation Context ===")
    for edge in actuation_edges:
        print(f"\nEdge: {edge[0]} -> {edge[1]}")
        try:
            context = parser.get_actuation_context(edge)
            
            print(f"Target: {context['target_concept']}")
            print(f"Required actuation: {context['as_target']['required_actuation']}")
            print(f"Required perception: {sorted(context['as_target']['required_perception'])}")
            print(f"Actuates: {sorted(context['as_source']['actuation_source_for'])}")
            print(f"Perceived by: {sorted(context['as_source']['perception_source_for'])}")
            
            # Validate context
            assert context['target_concept'] == edge[1], "Target concept should match edge destination"
            assert context['as_target']['required_actuation'] == edge[0], "Required actuation should match edge source"
            assert isinstance(context['as_target']['required_perception'], set), "Required perception should be a set"
            assert isinstance(context['as_source']['actuation_source_for'], set), "Actuation source for should be a set"
            assert isinstance(context['as_source']['perception_source_for'], set), "Perception source for should be a set"
            
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("=== Testing Related Concepts ===")
    test_get_related_concepts()
    
    print("\n=== Testing Actuation Context ===")
    test_actuation_context() 