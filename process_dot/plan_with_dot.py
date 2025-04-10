import re
import json
import ast
import os
from typing import Dict, List, Set, Optional, Union, Any

from normalign_stereotype.core._plan import Plan
from normalign_stereotype.core._agent import Agent
from normalign_stereotype.core._concept import create_concept_reference
from normalign_stereotype.core._modified_llm import ConfiguredLLM, BulletLLM, StructuredLLM
from normalign_stereotype.core._inference import Inference
from normalign_stereotype.core._reference import Reference
from normalign_stereotype.core._inference import Inference

class DOTParser:
    def __init__(self, dot_file_path: str) -> None:
        """Initialize DOT parser with path to DOT file.
        
        Args:
            dot_file_path: Path to the DOT file to parse
        """
        if not os.path.exists(dot_file_path):
            raise FileNotFoundError(f"DOT file not found: {dot_file_path}")
        self.dot_file_path = dot_file_path
        self.node_labels: Dict[str, List[str]] = {}
        self.edges: List[tuple[str, str, str]] = []
        self.concepts: Set[str] = set()
        self.classifications: Set[str] = set()
        self.base_concepts: Set[str] = set()
        
    def parse(self) -> None:
        """Parse the DOT file and extract nodes, edges, and concepts."""
        try:
            with open(self.dot_file_path, 'r') as f:
                content = f.read()
        except IOError as e:
            raise IOError(f"Failed to read DOT file: {e}")
            
        # Extract nodes and their labels
        node_pattern = r'(\w+)\s*\[xlabel="([^"]+)"\];'
        nodes = re.findall(node_pattern, content)
        for node, label in nodes:
            try:
                # Remove any whitespace and convert string representation to actual list
                label = label.strip()
                if label.startswith('{') and label.endswith('}'):
                    # Convert set-like notation to list
                    label = label[1:-1]  # Remove curly braces
                    label = label.replace("'", "")  # Remove quotes
                    label = [item.strip() for item in label.split(',') if item.strip()]
                else:
                    # Handle other formats if needed
                    label = ast.literal_eval(label)
                self.node_labels[node] = label
            except (SyntaxError, ValueError) as e:
                print(f"Warning: Could not parse label for node {node}: {label}")
                self.node_labels[node] = []
                
            if '_classification' in node:
                self.classifications.add(node)
            else:
                self.concepts.add(node)
                
        # Extract edges
        edge_pattern = r'(\w+)\s*->\s*(\w+)\s*\[label="(\w+)"\]'
        self.edges = re.findall(edge_pattern, content)
        
        # Identify base concepts (those without perception dependencies)
        for concept in self.concepts:
            if not self._get_concept_dependencies(concept):
                self.base_concepts.add(concept)
        
    def _get_concept_dependencies(self, concept: str) -> Set[str]:
        """Get all concepts that a given concept depends on.
        
        Args:
            concept: The concept to get dependencies for
            
        Returns:
            Set of concept names that the given concept depends on
        """
        dependencies = set()
        for src, dst, label in self.edges:
            if dst == concept and label == 'perc':
                dependencies.add(src)
        return dependencies
    
    def _get_actuation_concept(self, concept: str) -> Optional[str]:
        """Get the actuation concept for a given concept.
        
        Args:
            concept: The concept to get actuation for
            
        Returns:
            Name of the actuation concept, or None if not found
        """
        for src, dst, label in self.edges:
            if dst == concept and label == 'actu':
                return src
        return None


def create_plan_from_dot(dot_file_path: str, 
                        model_name: str = 'qwen-turbo-latest', 
                        reference_dir: str = "normalign_stereotype/concepts/stereotype_concepts",
                        input_concepts: Optional[Union[str, List[str]]] = 'statements', 
                        output_concept: str = 'answers') -> Plan:
    """Create a plan from a DOT file with specified input and output concepts.
    
    Args:
        dot_file_path: Path to the DOT file
        model_name: Name of the model to use
        reference_dir: Directory containing reference files
        input_concepts: List of input concept names or single input concept name. 
                       If None, uses base concepts
        output_concept: Name of the output concept
        
    Returns:
        A configured Plan object
        
    Raises:
        FileNotFoundError: If the DOT file or reference directory doesn't exist
        ValueError: If input or output concepts are invalid
        IOError: If there's an error reading or writing files
    """
    # Validate reference directory
    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
    
    # Initialize agent with memory file
    memory_file = "memory.json"
    # Clean memory file if it exists before creating new one
    if os.path.exists(memory_file):
        try:
            os.remove(memory_file)
        except Exception:
            pass  # Ignore cleanup errors if file is locked or doesn't exist
            
    try:
        with open(memory_file, "w") as f:
            json.dump({}, f)
    except IOError as e:
        raise IOError(f"Failed to initialize memory file: {e}")
        
    try:
        body = {
            "llm": ConfiguredLLM(model_name),
            "structured_llm": StructuredLLM(model_name),
            "bullet_llm": BulletLLM(model_name),
            "memory_location": memory_file
        }
        agent = Agent(body)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize agent: {e}")
    
    # Parse DOT file
    parser = DOTParser(dot_file_path)
    parser.parse()
    
    # Create plan
    plan = Plan(agent)
    
    # Add all concepts
    for concept in parser.concepts:
        plan.add_concept(concept)
    
    # Add all classification concepts
    for classification in parser.classifications:
        plan.add_concept(classification)
    
    # Configure I/O
    if input_concepts is None:
        input_concepts = list(parser.base_concepts)
    elif isinstance(input_concepts, str):
        input_concepts = [input_concepts]
    
    # Validate input concepts
    for concept in input_concepts:
        if concept not in parser.concepts:
            raise ValueError(f"Input concept '{concept}' not found in DOT file")
    
    # Validate output concept
    if output_concept not in parser.concepts:
        raise ValueError(f"Output concept '{output_concept}' not found in DOT file")
        
    plan.configure_io(input_names=input_concepts, output_name=output_concept)
    
    # Add inferences
    for concept in parser.concepts:
        if concept in parser.base_concepts:
            continue
            
        # Get perception concepts
        perception_concepts = list(parser._get_concept_dependencies(concept))
        
        # Get actuation concept
        actuation_concept = parser._get_actuation_concept(concept)
        if not actuation_concept:
            continue
            
        # Get view from node label
        view = parser.node_labels[concept]
        
        # Add inference
        plan.add_inference(
            perception_concept_names=perception_concepts,
            actuation_concept_name=actuation_concept,
            inferred_concept_name=concept,
            view=view
        )
    
    
    # Load references and make references for non-input base concepts and classification concepts
    for concept in parser.base_concepts + parser.classifications:
        if concept in input_concepts:
            continue

        # make reference 
        try:
            file_path = os.path.join(reference_dir, concept)
            plan.make_reference(concept, reference_path=file_path, read_reference=True)
        except Exception as e:
            print(f"Warning: Could not load reference for concept {concept}: {e}")
            # Fallback to using the concept name as a value if reference loading fails
            plan.make_reference(concept, reference=create_concept_reference(concept, concept, concept), read_reference=False)


    return plan

if __name__ == "__main__":
    try:
        # Example usage with custom input and output concepts
        dot_file = "process_dot/stereotype_graphvis_draft.dot"
        input_concepts = {"statements": "you are funny"} # Specify your input concepts
        output_concept = "answers"  # Specify your output concept
        
        plan = create_plan_from_dot(
            dot_file,
            reference_dir="normalign_stereotype/concepts/stereotype_concepts",
            input_concepts=input_concepts.keys(),
            output_concept=output_concept
        )

        # for input concepts, make references
        input_references = {}
        for concept, value in input_concepts.items():
            input_references[concept] = create_concept_reference(concept, value)
        
        answer_ref = plan.execute(input_references)
        print("\nFinal Inference Results:")
        print("Reference Tensor:", answer_ref.tensor)
        print("Tensor Axes:", answer_ref.axes)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except ValueError as e:
        print(f"Error: Invalid configuration - {e}")
    except IOError as e:
        print(f"Error: I/O operation failed - {e}")
    except Exception as e:
        print(f"Error executing plan: {e}") 