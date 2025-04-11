from normalign_stereotype.core._concept import Concept, create_concept_reference
from normalign_stereotype.core._agent import Agent, get_default_working_config
from normalign_stereotype.core._inference import Inference
from normalign_stereotype.core._reference import Reference


from typing import Optional, Any, Dict, List
from collections import defaultdict, deque
import ast

class Plan:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.concept_registry: Dict[str, Concept] = {}
        self.inference_registry: Dict[str, Inference] = {}
        self.inference_order: List[Inference] = []
        self.input_concept_names: List[str] = []
        self.output_concept_name: Optional[str] = None

    def configure_io(self, input_names, output_name):
        for name in input_names + [output_name]:
            if name not in self.concept_registry:
                raise ValueError(f"Concept '{name}' not registered")

        self.input_concept_names = input_names
        self.output_concept_name = output_name
        return self

    def add_concept(self, concept_name):
        concept = Concept(concept_name)
        self.concept_registry[concept_name] = concept
        return concept

    def make_reference(self, concept_name, reference: Reference = None, reference_path = None, actuation_working_config=None, read_reference=True):
        concept = self.concept_registry[concept_name]
        if read_reference:
            concept.read_reference_from_file(reference_path)
        else:
            concept.reference = reference
        
        # Get default config
        perception_config, actuation_config = get_default_working_config(concept_name)
        
        # Apply custom actuation if provided
        if actuation_working_config:
            actuation_config = actuation_working_config
        
        # Execute cognition with custom configuration
        concept.reference = self.agent.cognition(
            concept_name,
            perception_working_config=self.perception_config,
            actuation_working_config=self.actuation_config
        )
        return self

    def add_inference(self, perception_concept_names, actuation_concept_name, inferred_concept_name, view=None, actuation_working_config=None, perception_working_config=None):
        """Now includes optional view configuration and registry tracking"""
        # Validate concepts exist
        for name in perception_concept_names + [actuation_concept_name, inferred_concept_name]:
            if name not in self.concept_registry:
                self.add_concept(name)  # Auto-create if not exists (or raise error if preferred)

        perception_concepts = [self.concept_registry[name] for name in perception_concept_names]
        actuation_concept = self.concept_registry[actuation_concept_name]
        inferred_concept = self.concept_registry[inferred_concept_name]

        # Create inference key
        inference_key = str([perception_concept_names, actuation_concept_name, inferred_concept_name])

        # Check for duplicate inference
        if inference_key in self.inference_registry:
            raise ValueError(f"Inference {inference_key} already exists")

        inference = Inference(inferred_concept, self.agent)

        if view:
            inference.view_definition(view)

        inference.inference_definition(
            perception_concepts=perception_concepts,
            actuation_concept=actuation_concept,
            perception_working_config=perception_working_config if perception_working_config else None,
            actuation_working_config=actuation_working_config if actuation_working_config else None
        )
        
        # Store in both registry
        self.inference_registry[inference_key] = inference
        return self

    def order_inference(self):
        # 1. Identify foundational concepts (inputs + referenced)
        initial_concepts = set(self.input_concept_names)
        initial_concepts.update(
            name for name, concept in self.concept_registry.items()
            if hasattr(concept, 'reference') and concept.reference is not None
        )

        # 2. Build concept production map and reverse mapping
        concept_producers = {}  # concept_name -> producer inference
        inf_to_components = {}  # inference -> (input_concepts, output_concept)

        # Create reverse mapping from inferences to their registry keys
        inf_to_key = {v: k for k, v in self.inference_registry.items()}

        # Parse all registry keys to get component relationships
        for inf in self.inference_registry.values():
            key = inf_to_key[inf]
            components = ast.literal_eval(key)
            perception_names, actuation_name, inferred_name = components

            # Validate concept relationships
            if inf.concept_to_infer.comprehension["name"] != inferred_name:
                raise ValueError(f"Inference registry mismatch for {inf}")

            # Store input/output relationships
            input_concepts = set(perception_names) | {actuation_name}
            inf_to_components[inf] = (input_concepts, inferred_name)

            # Register concept producer
            if inferred_name in concept_producers:
                raise ValueError(f"Multiple producers for {inferred_name}")
            concept_producers[inferred_name] = inf

        # 3. Build dependency graph using registry data
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        for inf in self.inference_registry.values():
            input_concepts, _ = inf_to_components[inf]

            # Find dependencies that require prior inferences
            dependencies = set()
            for concept in input_concepts:
                if concept not in initial_concepts:
                    if concept not in concept_producers:
                        raise ValueError(f"Unresolvable dependency: {concept}")
                    dependencies.add(concept_producers[concept])

            # Create graph edges
            for dep_inf in dependencies:
                graph[dep_inf].append(inf)
                in_degree[inf] += 1

            # Initialize nodes with no dependencies
            if inf not in in_degree:
                in_degree[inf] = 0

        # 4. Kahn's algorithm with initial roots
        queue = deque([
            inf for inf in self.inference_registry.values()
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

        # 5. Validate and update pipeline
        if len(ordered) != len(self.inference_registry):
            remaining = set(self.inference_registry.values()) - set(ordered)
            cycle_info = [
                f"{inf_to_components[inf][1]} (requires {inf_to_components[inf][0]})"
                for inf in remaining
            ]
            raise ValueError(
                f"Cyclic/missing dependencies detected in: {cycle_info}"
            )

        self.inference_order = ordered
        return self

    def execute(self, input_data: Optional[dict[str, Reference]] = None, input_config: Optional[dict[str, dict[str, dict]]] = None):
        """Execute the plan with optional input data, returning the output concept reference"""
        # Validate I/O configuration
        if not self.input_concept_names or not self.output_concept_name:
            raise ValueError("I/O not configured. Call configure_io() first")

        # Process input data
        if input_data is not None:
            # Validate input format
            if not isinstance(input_data, dict):
                raise TypeError("Input data must be a dictionary")

            # Check all required inputs are present
            missing_inputs = set(self.input_concept_names) - set(input_data.keys())
            if missing_inputs:
                raise ValueError(f"Missing input data for: {', '.join(missing_inputs)}")

            # Set input concept references
            for name in self.input_concept_names:
                concept = self.concept_registry[name]
                concept.reference = input_data[name]
                # Get default config and execute
                self.make_reference(
                    concept_name=name,
                    reference=input_data[name],
                    actuation_working_config=input_config[name]["actuation"],
                    read_reference=False
                )
        else:
            # Verify preconfigured references exist
            missing_refs = [
                name for name in self.input_concept_names
                if not self.concept_registry[name].reference
            ]
            if missing_refs:
                raise ValueError(
                    f"Missing references for inputs: {', '.join(missing_refs)}. "
                    "Either provide input_data or use make_reference()"
                )

        # Execute inference_order in topological order
        if not self.inference_order:
            self.order_inference()

        for inf in self.inference_order:
            inf.execute()

        # Retrieve and validate final output
        output_concept = self.concept_registry[self.output_concept_name]

        if not output_concept.reference:
            raise RuntimeError(
                f"Output concept '{self.output_concept_name}' "
                "failed to generate a reference"
            )

        return output_concept.reference
