from normalign_stereotype.core._llm_tools import ConfiguredLLM, BulletLLM, StructuredLLM
from normalign_stereotype.core._objects._reference import Reference, cross_action, cross_product, element_action
from normalign_stereotype.core._objects._concept import Concept
from normalign_stereotype.core._agent_frame import Agent
from normalign_stereotype.core._objects._inference import Inference

import json
import os


if __name__ == "__main__":

    """create agent with memory and body"""

    with open("../memory.json", "w") as f:
        json.dump({}, f)  # Initialize empty JSON
    body = { # Create the 'body' dictionary
        "llm": ConfiguredLLM(),  # Primary LLM (required)
        "structured_llm": StructuredLLM(),  # Optional (for structured tasks)
        "bullet_llm": BulletLLM() ,  # Optional (for bullet-point tasks)
        "memory_location": "memory.json",  # Path to memory file (required)
    }
    agent = Agent(body)

    """initiate the target group and individuals classification concept"""

    concept_name = "target_group"
    target_group_concept = Concept(concept_name)
    target_group_concept.read_reference_from_file(f"stereotype_concepts/{concept_name}_ref")
    target_group_concept.reference = Inference(target_group_concept, agent).cognition_configuration()

    concept_name = "individuals_classification"
    individuals_classification_concept = Concept(concept_name)
    individuals_classification_concept.read_reference_from_file(f"stereotype_concepts/{concept_name}_ref")
    individuals_classification_concept.reference = Inference(individuals_classification_concept, agent).cognition_configuration()

    print(agent.working_memory)

    """start inference"""

    # Create inference concept to receive results
    individuals_concept = Concept("individuals")

    # Set up and execute inference pipeline
    inference = Inference(individuals_concept, agent)

    # Define the inference relationship
    inference.inference_definition(
        perception_concepts=[target_group_concept],
        actuation_concept=individuals_classification_concept
    )

    # Optional: Set view
    inference.view_definition(start="individuals_classification", end="individuals")

    # Execute the full inference process
    inference.execute()

    # Display results
    print("\nFinal Inference Results:")
    print("Concept Name:", individuals_concept.comprehension["name"])
    print("Reference Tensor:", individuals_concept.reference.tensor)
    print("Tensor Axes:", individuals_concept.reference.axes)

    # Inspect agent memory state
    print("\nAgent Memory State:")
    print("Working Memory:", agent.working_memory)
    print("Persisted Memory:", json.dumps(json.load(open("../memory.json")), indent=2))

    print("\nTest Perception on View Change:")
    print(agent.perception(individuals_concept).tensor)
    print("Tensor Axes:", agent.perception(individuals_concept).axes)