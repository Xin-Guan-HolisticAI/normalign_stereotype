from normalign_stereotype.core._objects._concept import Concept
from normalign_stereotype.core._objects._reference import Reference
from typing import Optional, List, Union, Dict, Set, Tuple, Any
import logging
from normalign_stereotype.core._objects._inference import Inference
from normalign_stereotype.core._objects._utils import (
    _get_initial_concepts,
    _build_concept_mappings,
    _build_dependency_graph,
    _topological_sort,
    _validate_topological_order,
    _process_input_data
)


class Plan:
    def __init__(self, debug: bool = False):
        self.agent: Optional[Any] = None
        self.concept_registry: Dict[str, Concept] = {}
        self.inference_registry: Dict[str, Inference] = {}
        self.inference_order: List[Inference] = []
        self.input_concept_names: List[str] = []
        self.constant_concept_names: List[str] = []
        self.output_concept_name: Optional[str] = None
        self.debug = debug

    def _debug_print(self, message: str):
        if self.debug:
            print(f"[DEBUG] {message}")

    def add_concept(self, concept: Optional[Concept] = None, **kwargs):
        if concept is None:
            concept = Concept(**kwargs)
        self.concept_registry[concept.comprehension["name"]] = concept
        return concept

    def add_inference(self, inference: Optional[Inference] = None, **kwargs):
        if inference is None:
            inference = Inference(**kwargs)
        perception_concepts = [c.comprehension["name"] for c in inference.post_cognition_pre_perception_concepts]
        actuation_concept = inference.post_cognition_pre_actuation_concept.comprehension["name"]
        inferred_concept = inference.concept_to_infer.comprehension["name"]
        inference_key = str([perception_concepts, actuation_concept, inferred_concept])
        self.inference_registry[inference_key] = inference
        return inference
    
    def set_input_output_concepts(self, input_names, output_name):
        for name in input_names + [output_name]:
            if name not in self.concept_registry:
                raise ValueError(f"Concept '{name}' not registered")

        self.input_concept_names = input_names
        self.output_concept_name = output_name
        return self
    
    def direct_reference_to_concept(self, agent, input_mode: str = "raw", input_data: Optional[dict[str, Union[Reference, str, dict]]] = None, 
                input_config: Optional[dict[str, dict[str, dict]]] = None):
        """Process constants and update the inference registry."""
        from normalign_stereotype.core._agent_frame._agent_main import AgentFrame
        if not isinstance(agent, AgentFrame):
            raise ValueError("Agent must be an instance of AgentFrame")
        self.agent = agent
        self._debug_print("Starting direct reference.")

        # Check if all constant concepts are registered
        unregistered_concepts = [name for name in input_data.keys() if name not in self.concept_registry.keys()]
        if unregistered_concepts:
            raise ValueError(f"Some constant concepts are not registered: {', '.join(unregistered_concepts)}")

        # Add constant concepts to the list
        self.constant_concept_names.extend([
            name for name in input_data.keys()
            if name not in self.constant_concept_names
        ])

        # Process constants and get updated concepts
        processed_concepts = _process_input_data(
            input_mode, input_data, input_config, 
            self.constant_concept_names, self.concept_registry, 
            agent, True, self.debug
        )
        
        # Update concept registry with processed concepts
        for concept in processed_concepts:
            self.concept_registry[concept.comprehension["name"]] = concept
            
        # Verify all constants have references
        missing_refs = [
            name for name in self.constant_concept_names
            if not self.concept_registry[name].reference
        ]
        if missing_refs:
            raise ValueError(f"Missing references for constants: {', '.join(missing_refs)}")

        # Renew relevant inferences
        for inf_key, inference in self.inference_registry.items():
            # Check if inference uses any of the constant concepts
            if any(concept.comprehension["name"] in self.constant_concept_names 
                  for concept in inference.post_cognition_pre_perception_concepts) or \
               inference.post_cognition_pre_actuation_concept.comprehension["name"] in self.constant_concept_names:
                # Renew the inference with updated concepts
                perception_concepts = [
                    self.concept_registry[c.comprehension["name"]] 
                    for c in inference.post_cognition_pre_perception_concepts
                ]
                actuation_concept = self.concept_registry[
                    inference.post_cognition_pre_actuation_concept.comprehension["name"]
                ]
                inferred_concept = self.concept_registry[
                    inference.concept_to_infer.comprehension["name"]
                ]
                
                self.inference_registry[inf_key] = Inference(
                    concept_to_infer=inferred_concept,
                    perception_concepts=perception_concepts,
                    actuation_concept=actuation_concept,
                    view=inference.view
                )

        return self

    def order_inference(self):
        """Order inferences based on their dependencies using topological sorting."""
        # 1. Get initial concepts
        initial_concepts = _get_initial_concepts(self.input_concept_names, self.concept_registry)

        # 2. Build concept mappings
        concept_producers, inf_to_components, _ = _build_concept_mappings(self.inference_registry)

        # 3. Build dependency graph
        graph, in_degree = _build_dependency_graph(
            initial_concepts, concept_producers, inf_to_components, self.inference_registry
        )

        # 4. Perform topological sort
        ordered = _topological_sort(graph, in_degree, self.inference_registry)

        # 5. Validate and update pipeline
        _validate_topological_order(ordered, inf_to_components, self.inference_registry)

        self.inference_order = ordered
        return self

    def execute(self, agent, input_mode: str = "raw", input_data: Optional[dict[str, Union[Reference, str, dict]]] = None, 
                input_config: Optional[dict[str, dict[str, dict]]] = None):
        from normalign_stereotype.core._agent_frame._agent_main import AgentFrame
        if not isinstance(agent, AgentFrame):
            raise ValueError("Agent must be an instance of AgentFrame")
        self.agent = agent
        self._debug_print("Starting plan execution")

        # Validate I/O configuration
        if not self.input_concept_names or not self.output_concept_name:
            raise ValueError("I/O not configured. Call configure_io() first")

        # Process input data and get updated concepts
        processed_concepts = _process_input_data(
            input_mode, input_data, input_config, 
            self.input_concept_names, self.concept_registry, 
            agent, True, self.debug
        )
        
        # Update concept registry with processed concepts
        for concept in processed_concepts:
            self.concept_registry[concept.comprehension["name"]] = concept

        # Verify all input concepts have references
        missing_refs = [
            name for name in self.input_concept_names
            if not self.concept_registry[name].reference
        ]
        if missing_refs:
            raise ValueError(f"Missing references for inputs: {', '.join(missing_refs)}")

        # Renew relevant inferences
        for inf_key, inference in self.inference_registry.items():
            # Check if inference uses any of the input concepts
            if any(concept.comprehension["name"] in self.input_concept_names 
                  for concept in inference.post_cognition_pre_perception_concepts) or \
               inference.post_cognition_pre_actuation_concept.comprehension["name"] in self.input_concept_names:
                # Renew the inference with updated concepts
                perception_concepts = [
                    self.concept_registry[c.comprehension["name"]] 
                    for c in inference.post_cognition_pre_perception_concepts
                ]
                actuation_concept = self.concept_registry[
                    inference.post_cognition_pre_actuation_concept.comprehension["name"]
                ]
                inferred_concept = self.concept_registry[
                    inference.concept_to_infer.comprehension["name"]
                ]
                
                self.inference_registry[inf_key] = Inference(
                    concept_to_infer=inferred_concept,
                    perception_concepts=perception_concepts,
                    actuation_concept=actuation_concept,
                    view=inference.view
                )

        # Execute inference_order in topological order
        if not self.inference_order:
            self.order_inference()

        self._debug_print("Executing inferences in order:")
        for i, inf in enumerate(self.inference_order):
            self._debug_print(f"  {i+1}. Executing inference for {inf.concept_to_infer.comprehension['name']}")
            inf.execute(agent=agent)

        # Retrieve and validate final output
        output_concept = self.concept_registry[self.output_concept_name]
        self._debug_print(f"Retrieving output from {self.output_concept_name}")

        if not output_concept.reference:
            raise RuntimeError(
                f"Output concept '{self.output_concept_name}' "
                "failed to generate a reference"
            )

        return output_concept.reference
