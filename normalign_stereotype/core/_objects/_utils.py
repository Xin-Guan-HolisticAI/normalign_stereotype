"""
Utility functions for the NPC (Neural Processing Core) package.

This module contains helper functions used across the NPC package for processing
concepts, building dependency graphs, and handling input data.
"""

from typing import Optional, List, Union, Dict, Set, Tuple, Any
from collections import defaultdict
from collections import deque
import ast
import json
from string import Template
import logging

from normalign_stereotype.core._objects._concept import Concept
from normalign_stereotype.core._objects._reference import Reference
from normalign_stereotype.core._objects._inference import Inference

def _get_initial_concepts(input_concept_names: List[str], concept_registry: Dict[str, Any]) -> Set[str]:
    """Get the set of initial concepts (inputs + referenced concepts)."""
    initial_concepts = set(input_concept_names)
    initial_concepts.update(
        name for name, concept in concept_registry.items()
        if hasattr(concept, 'reference') and concept.reference is not None
    )
    return initial_concepts

def _build_concept_mappings(inference_registry: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[Any, Tuple[Set[str], str]], Dict[Any, str]]:
    """Build mappings between inferences and their components.
    
    Returns:
        tuple containing:
        - concept_producers: dict mapping concept names to their producer inferences
        - inf_to_components: dict mapping inferences to their (input_concepts, output_concept)
        - inf_to_key: dict mapping inferences to their registry keys
    """
    concept_producers = {}
    inf_to_components = {}
    inf_to_key = {v: k for k, v in inference_registry.items()}

    for inf in inference_registry.values():
        key = inf_to_key[inf]
        components = ast.literal_eval(key)
        perception_names, actuation_name, inferred_name = components

        if inf.concept_to_infer.comprehension["name"] != inferred_name:
            raise ValueError(f"Inference registry mismatch for {inf}")

        input_concepts = set(perception_names) | {actuation_name}
        inf_to_components[inf] = (input_concepts, inferred_name)

        if inferred_name in concept_producers:
            raise ValueError(f"Multiple producers for {inferred_name}")
        concept_producers[inferred_name] = inf

    return concept_producers, inf_to_components, inf_to_key

def _build_dependency_graph(initial_concepts: Set[str], concept_producers: Dict[str, Any], 
                          inf_to_components: Dict[Any, Tuple[Set[str], str]],
                          inference_registry: Dict[str, Any]) -> Tuple[defaultdict, defaultdict]:
    """Build the dependency graph for topological sorting.
    
    Args:
        initial_concepts: Set of initial concept names
        concept_producers: Dict mapping concept names to their producer inferences
        inf_to_components: Dict mapping inferences to their (input_concepts, output_concept)
        inference_registry: Dict containing all registered inferences
        
    Returns:
        tuple containing:
        - graph: adjacency list representation of the dependency graph
        - in_degree: dict mapping nodes to their in-degree
    """
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for inf in inference_registry.values():
        input_concepts, _ = inf_to_components[inf]
        dependencies = set()

        for concept in input_concepts:
            if concept not in initial_concepts:
                if concept not in concept_producers:
                    raise ValueError(f"Unresolvable dependency: {concept}")
                dependencies.add(concept_producers[concept])

        for dep_inf in dependencies:
            graph[dep_inf].append(inf)
            in_degree[inf] += 1

        if inf not in in_degree:
            in_degree[inf] = 0

    return graph, in_degree

def _topological_sort(graph: defaultdict, in_degree: defaultdict, 
                     inference_registry: Dict[str, Any]) -> List[Any]:
    """Perform topological sort using Kahn's algorithm.
    
    Args:
        graph: Adjacency list representation of the dependency graph
        in_degree: Dict mapping nodes to their in-degree
        inference_registry: Dict containing all registered inferences
        
    Returns:
        List of inferences in topological order
    """
    queue = deque([
        inf for inf in inference_registry.values()
        if in_degree[inf] == 0
    ])
    ordered = []

    while queue:
        current = queue.popleft()
        ordered.append(current)

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return ordered

def _validate_topological_order(ordered: List[Any], inf_to_components: Dict[Any, Tuple[Set[str], str]], 
                              inference_registry: Dict[str, Any]) -> None:
    """Validate that the topological sort includes all inferences."""
    if len(ordered) != len(inference_registry):
        remaining = set(inference_registry.values()) - set(ordered)
        cycle_info = [
            f"{inf_to_components[inf][1]} (requires {inf_to_components[inf][0]})"
            for inf in remaining
        ]
        raise ValueError(
            f"Cyclic/missing dependencies detected in: {cycle_info}"
        )

def _process_input_data(input_mode: str, input_data: Optional[dict[str, Union[Reference, str, dict]]], 
                       input_config: Optional[dict[str, dict[str, dict]]],
                       input_concept_names: List[str], concept_registry: Dict[str, Concept], 
                       agent: Optional[Any] = None, cognition_input: bool = True, debug: bool = False) -> List[Concept]:
    """Process input data and set up concept references.
    
    Returns:
        List of processed concepts with references set
    """
    from normalign_stereotype.core._agent_frame import AgentFrame, _create_concept_reference, _prompt_template_dynamic_substitution
    
    if agent is not None and not isinstance(agent, AgentFrame):
        raise ValueError("Agent must be an instance of AgentFrame")
        
    processed_concepts = []
    
    if debug:
        print(f"[DEBUG] Processing input data in mode: {input_mode}")
        print(f"[DEBUG] Input concepts: {input_concept_names}")

    if input_data is not None:
        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary")

        # Check all required inputs are present
        missing_inputs = set(input_concept_names) - set(input_data.keys())
        if missing_inputs:
            raise ValueError(f"Missing input data for: {', '.join(missing_inputs)}")

        # Process each input concept
        for name in input_concept_names:
            if debug:
                print(f"[DEBUG] Processing concept: {name}")
            
            concept = concept_registry[name]
            input_value = input_data[name]
            working_config = input_config[name] if input_config else {}
            
            if input_mode.startswith("raw"):
                mode_of_explanation = input_mode.split("_", 1)[1] if len(input_mode.split("_")) > 1 else "direct_explanation"

                if mode_of_explanation == "direct_explanation":
                    if isinstance(input_value, str):
                        if debug:
                            print(f"[DEBUG] Converting string input to dict for {name}")
                            print(f"[DEBUG] Input string: {input_value}")
                        try:
                            input_value = json.loads(input_value)
                            if isinstance(input_value, list):
                                input_value = input_value[0]
                        except json.JSONDecodeError as e:
                            if debug:
                                print(f"[DEBUG] JSON parsing failed, trying eval: {e}")
                            # If JSON parsing fails, try eval
                            try:
                                input_value = eval(input_value)
                                if isinstance(input_value, list):
                                    input_value = input_value[0]
                            except Exception as e:
                                if debug:
                                    print(f"[DEBUG] Eval failed: {e}")
                                raise ValueError(f"Failed to parse input for {name}: {e}")

                    if isinstance(input_value, dict):
                        if "Explanation" not in input_value:
                            raise ValueError(f"Raw input for {name} must contain 'Explanation'")
                        if "Summary_Key" not in input_value:
                            raise ValueError(f"Raw input for {name} must contain 'Summary_Key'")
                        reference_explanation = input_value["Explanation"]
                        reference_summary_key = input_value["Summary_Key"]
                    else:
                        raise ValueError(f"Raw input for {name} must be a dictionary")
                elif mode_of_explanation == "empty_explanation":
                    reference_explanation = ""
                    reference_summary_key = str(input_value)
                elif mode_of_explanation == "template_explanation":
                    template = working_config.get("template", "$input_value")
                    if isinstance(template, str):
                        template = Template(template)
                    elif isinstance(template, Template):
                        pass
                    else:   
                        raise ValueError(f"Template must be a string or Template")

                    reference_explanation = _prompt_template_dynamic_substitution(
                        prompt_template=template, 
                        template_variables={"input_value": input_value, 
                                            "concept_name": concept.comprehension["name"],
                                            "concept_context": concept.context,
                                            "concept_type": concept.type},
                    )
                    reference_summary_key = str(input_value)
                elif mode_of_explanation == "agent_explanation":
                    assert agent is not None
                    template = working_config.get("template", "What does $input_value likely mean? Explain in few sentences.")
                    prompt = _prompt_template_dynamic_substitution(
                        prompt_template=template, 
                        template_variables={"input_value": input_value, 
                                            "concept_name": concept.comprehension["name"],
                                            "concept_context": concept.context,
                                            "concept_type": concept.type},
                    )
                    llm_name = working_config.get("llm", "llm")
                    llm = agent.body[llm_name] 
                    #assert that llm has an invoke method
                    assert hasattr(llm, "invoke")

                    reference_explanation = llm.invoke(prompt)
                    reference_summary_key = str(input_value)
                else:
                    raise ValueError(f"Invalid mode of explanation: {mode_of_explanation}")

                if debug:
                    print(f"[DEBUG] Creating reference for {name}")
                    print(f"[DEBUG] Explanation: {reference_explanation}")
                    print(f"[DEBUG] Summary Key: {reference_summary_key}")
                
                # Create reference from raw input
                concept.reference = _create_concept_reference(
                    mode_of_remember=agent.working_memory["cognition"]["mode_of_remember"],
                    concept_name=concept.comprehension["name"],
                    explanation=reference_explanation,
                    summary_key=reference_summary_key,
                )

                #the concept must be go through the cognition process of the agent
                if cognition_input:
                    cognition_kwargs = {}
                    if "perception" in working_config:
                        cognition_kwargs["perception_working_config"] = working_config["perception"]
                    if "actuation" in working_config:
                        cognition_kwargs["actuation_working_config"] = working_config["actuation"]
                    concept.reference = agent.cognition(
                        concept=concept,
                        **cognition_kwargs
                    )   

                processed_concepts.append(concept)
    
    return processed_concepts 