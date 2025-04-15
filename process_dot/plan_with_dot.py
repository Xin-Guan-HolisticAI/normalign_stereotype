import re
import json
import ast
import os
from typing import Dict, List, Set, Optional, Union, Any

from normalign_stereotype.core._plan import Plan
from normalign_stereotype.core._agent import Agent, customize_actuation_working_config
from normalign_stereotype.core._concept import create_concept_reference
from normalign_stereotype.core._modified_llm import ConfiguredLLM, BulletLLM, StructuredLLM
from normalign_stereotype.core._inference import Inference
from normalign_stereotype.core._reference import Reference


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
        self.concept_names: Set[str] = set()
        self.classification_concept_names: Set[str] = set()
        self.base_concept_names: Set[str] = set()
        self.context: str = ""
        
    def parse(self) -> None:
        """Parse the DOT file and extract nodes, edges, and concepts."""
        try:
            with open(self.dot_file_path, 'r') as f:
                content = f.read()
                print("\n=== Raw content ===")
                print(content)
                
                # Extract context if present
                context_match = re.match(r'^###(.*?)(?=digraph|$)', content, re.DOTALL)
                if context_match:
                    self.context = context_match.group(1).strip()
                    print("\n=== Found context ===")
                    print(self.context)
                    # Remove context from content for further processing
                    content = content[context_match.end():].strip()
        except IOError as e:
            raise IOError(f"Failed to read DOT file: {e}")
            
        # Extract nodes and their labels
        node_pattern = r'\s*"([^"]+)"\s*\[xlabel\s*=\s*"([^"]+)"\](?:\s*;)?'
        nodes = re.findall(node_pattern, content)
        print("\n=== Found nodes ===")
        for node, label in nodes:
            print(f"Node: {node}")
            print(f"Label: {label}")
            print("---")
        
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
                print(f"\nProcessed node {node}")
                print(f"Final label: {label}")
            except (SyntaxError, ValueError) as e:
                print(f"Warning: Could not parse label for node {node}: {label}")
                self.node_labels[node] = []
                
            if ('_classification' in node) or ("?" in node):
                self.classification_concept_names.add(node)
            else:
                self.concept_names.add(node)
                
        # print("\n=== Concept Classification ===")
        # print("Regular concepts:", self.concept_names)
        # print("\nClassification concepts:", self.classification_concept_names)
        
        # Extract edges
        edge_pattern = r'"([^"]+)"\s*->\s*"([^"]+)"\s*\[label="(\w+)"\]'
        self.edges = re.findall(edge_pattern, content)
        # print("\n=== Found edges ===")
        for src, dst, label in self.edges:
            print(f"{src} --({label})--> {dst}")
        
        # Identify base concepts (those without perception dependencies)
        # print("\n=== Base Concept Analysis ===")
        for concept in self.concept_names:
            deps = self._get_concept_dependencies(concept)
            # print(f"\nConcept: {concept}")
            # print(f"Dependencies: {deps}")
            if not deps:
                self.base_concept_names.add(concept)
                # print("Added to base concepts!")
        
        # print("\n=== Final Results ===")
        # print("Base concepts:", self.base_concept_names)
        
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

    def get_related_concepts(self, concept: str) -> Dict[str, Set[str]]:
        """Get all related concepts for a given concept.
        
        Args:
            concept: The concept to get related concepts for
            
        Returns:
            Dictionary containing sets of related concepts with the following keys:
            - 'incoming_perception': Concepts that perceive this concept
            - 'incoming_actuation': Concepts that actuate this concept
            - 'outgoing_perception': Concepts that this concept perceives
            - 'outgoing_actuation': Concepts that this concept actuates
            
        Raises:
            ValueError: If the concept is not found in the parsed DOT file
        """
        if concept not in self.concept_names and concept not in self.classification_concept_names:
            raise ValueError(f"Concept '{concept}' not found in DOT file")
            
        related = {
            'incoming_perception': set(),
            'incoming_actuation': set(),
            'outgoing_perception': set(),
            'outgoing_actuation': set()
        }
        
        for src, dst, label in self.edges:
            if dst == concept and label == 'perc':
                related['incoming_perception'].add(src)
            elif dst == concept and label == 'actu':
                related['incoming_actuation'].add(src)
            elif src == concept and label == 'perc':
                related['outgoing_perception'].add(dst)
            elif src == concept and label == 'actu':
                related['outgoing_actuation'].add(dst)
                
        return related
    
    def get_actuation_context(self, edge: tuple[str, str, str]) -> Dict[str, Dict[str, Union[str, Set[str]]]]:
        """Get the context required for actuation.
        
        This method can be called in one way:
        1. With an actuation edge (src, dst, 'actu') to get context for that specific edge
        
        Args:
            edge: A tuple (source, destination, 'actu') representing an actuation edge
            
        Returns:
            Dictionary containing:
            - 'target_concept': The concept being actuated
            as_target:
            - 'required_actuation': The source concept of the actuation (input)
            - 'required_perception': Set of concepts that must be perceived by the target
            as_source:
            - 'actuation_source_for': The concepts that this target concept actuates
            - 'perception_source_for': The concepts that perceiving the target concept
            
        Raises:
            ValueError: If the edge is not a valid actuation edge or concepts are not found
        """
        src, dst, label = edge
        
        if label != 'actu':
            raise ValueError(f"Edge {edge} is not an actuation edge")
            
        if dst not in self.concept_names and dst not in self.classification_concept_names:
            raise ValueError(f"Target concept '{dst}' not found in DOT file")
            
        context = {
            'target_concept': dst,
            'as_target': {
                'required_actuation': src,
                'required_perception': self._get_concept_dependencies(dst)
            },
            'as_source': {
                'actuation_source_for': set(),
                'perception_source_for': set()
            }
        }
        
        # Get concepts that this target concept actuates
        for edge_src, edge_dst, edge_label in self.edges:
            if edge_src == dst and edge_label == 'actu':
                context['as_source']['actuation_source_for'].add(edge_dst)    
        # Get concepts that perceive this target concept
            if edge_src == dst and edge_label == 'perc':
                context['as_source']['perception_source_for'].add(edge_dst)
                
        return context



def create_plan_from_dot(dot_file_path: str, 
                        model_name: str = 'qwen-turbo-latest', 
                        reference_dir: str = "normalign_stereotype/concepts/stereotype_concepts",
                        working_config: Dict[str, Dict[str, Dict]] = {},
                        input_concepts: Optional[Union[str, List[str]]] = 'statements', 
                        output_concept: str = 'answers') -> Plan:
    """Create a plan from a DOT file with specified input and output concepts.
    
    Args:
        dot_file_path: Path to the DOT file
        model_name: Name of the model to use
        reference_dir: Directory containing reference files
        configuration: Dictionary of customized configuration for each concept
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
    for concept in parser.concept_names:
        plan.add_concept(concept, context=parser.context)
    
    # Add all classification concepts
    for classification in parser.classification_concept_names:
        plan.add_concept(classification, context=parser.context)
    
    # Configure I/O
    if input_concepts is None:
        input_concepts = list(parser.base_concept_names)
    elif isinstance(input_concepts, str):
        input_concepts = [input_concepts]
    
    # Validate input concepts
    for concept in input_concepts:
        if concept not in parser.concept_names:
            raise ValueError(f"Input concept '{concept}' not found in DOT file")
    
    # Validate output concept
    if output_concept not in parser.concept_names:
        raise ValueError(f"Output concept '{output_concept}' not found in DOT file")
        
    plan.configure_io(input_names=input_concepts, output_name=output_concept)
    
    # Add inferences
    for concept in parser.concept_names:
        if concept in parser.base_concept_names:
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
            view=view,
            actuation_working_config=working_config.get(concept, {}).get("actuation", None),
            perception_working_config=working_config.get(concept, {}).get("perception", None)
        )
        
    
    # Load references and make references for non-input base concepts and classification concepts
    for concept in parser.base_concept_names + parser.classification_concept_names:
        if concept in input_concepts:
            continue

        # make reference 
        try:
            file_path = os.path.join(reference_dir, concept)
            plan.make_reference(
                concept, 
                actuation_working_config=working_config.get(concept, {}).get("actuation", None),
                reference_path=file_path, 
                read_reference=True
                )
        except Exception as e:
            print(f"Warning: Could not load reference for concept {concept}: {e}")
            # Fallback to using the concept name as a value if reference loading fails
            plan.make_reference(
                concept, 
                actuation_working_config=working_config.get(concept, {}).get("actuation", None),
                reference=create_concept_reference(concept, concept, concept), 
                read_reference=False
                )


    return plan









if __name__ == "__main__":

    concept_comprehension_prompt = """
    
    Given the context: "{concept_context}"
    Explain what "{concept_name}" means, while keeping its meaning independent. Give a one-to-several sentences for definition, and make sure it is independent of the context.

    """


    classification_prompt = """

    Your task is to find instances of "{meta_name}".
     
    Context: "{meta_value}"
   
    Find from: "{input_value}"

    Your output should be some context and explanations following a summary name for each of the instance,


    """


    judgement_prompt = """

    Your task is to judge if the truth condition of "{meta_name}" given the variables are "{meta_name_variables}" being "{input_names}".

    That is to say, your task is to judge if "{meta_name_substituted}" is true.
    
    Context: 
    - "{meta_names}": "{meta_value}"
    {input_names_input_value_pairs}

    Your output should be some justifications following your judge i.e. "True", "False", or "Not Sure".

    """





    # def customize_reference_file(reference_dir, mode = "llm_with_default_template"):
    #     pass

    # # def customize_working_config_with_actuation_prompt_template(concept_name, concept_location, prompt_template_dir):
    # #     # prompt template
    # #     prompt_template_path = os.path.join(prompt_template_dir, f"{concept_name}.txt")


    # #     # read prompt template
    # #     with open(prompt_template_path, "w") as f:
    # #         prompt_template = f.read()

    # #     # build actuation prompt template
    # #     actuation_prompt_template = prompt_template.split("---")[0]

    # #     # build perception prompt template
    # #     perception_prompt_template = prompt_template.split("---")[1]


    # #     # build the actuation prompt template 



    # #     pass

    
    try:
        # Example usage with custom input and output concepts
        dot_file = "process_dot/metaphor_draft.dot"
        input_concepts = {"extract": "you are funny"} # Specify your input concepts
        # input_config = {
        #     "statements": customize_actuation_working_config(
        #         "statements",
        #         "process_dot/concepts/stereotype_concepts",
        #         mode="llm_prompt_two_replacement",
        #     )
        # }
        output_concept = "figurative_language_element_that_maps_specific_tangible_entity_onto_abstract_complex_theme_directly"
        
        parser = DOTParser(dot_file)
        parser.parse()

        # print(parser.base_concept_names)

        # for concept_name in parser.base_concept_names + parser.classification_concept_names:
        #     if concept_name in input_concepts.keys():
        #         continue

        #     customize_reference_file(
        #         reference_dir = "process_dot/concepts/stereotype_concepts"
        #     )

        # # customize working config
        # for concept_name in parser.concept_names:
        #     working_config = customize_working_config_with_actuation_prompt_template(
        #         prompt_template_dir = "process_dot/concepts/stereotype_concepts"
        #     )


        # plan = create_plan_from_dot(
        #     dot_file,
        #     reference_dir="process_dot/concepts/stereotype_concepts",
        #     # working_config=working_config,
        #     input_concepts=input_concepts.keys(),
        #     output_concept=output_concept
        # )

        # # for input concepts, make references
        # input_references = {}
        # for concept, value in input_concepts.items():
        #     input_references[concept] = create_concept_reference(concept, value)
        
        # answer_ref = plan.execute(input_references, input_config)
    #     print("\nFinal Inference Results:")
    #     print("Reference Tensor:", answer_ref.tensor)
    #     print("Tensor Axes:", answer_ref.axes)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    # except ValueError as e:
    #     print(f"Error: Invalid configuration - {e}")
    # except IOError as e:
    #     print(f"Error: I/O operation failed - {e}")
    # except Exception as e:
    #     print(f"Error executing plan: {e}") 