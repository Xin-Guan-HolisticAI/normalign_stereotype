from normalign_stereotype.core._modified_llm import ConfiguredLLM, BulletLLM, StructuredLLM
from normalign_stereotype.core._agent import Agent
from normalign_stereotype.core._plan import Plan
from normalign_stereotype.core._reference import Reference
import json

if __name__ == "__main__":
    # Initialize agent with memory and body
    with open("../memory.json", "w") as f:
        json.dump({}, f)  # Initialize empty memory
    body = {
        "llm": ConfiguredLLM(),
        "structured_llm": StructuredLLM(),
        "bullet_llm": BulletLLM(),
        "memory_location": "memory.json"
    }
    agent = Agent(body)

    # Create and configure plan
    plan = Plan(agent)

    # Register concepts and references
    plan.add_concept("target_group")
    plan.make_reference("target_group", "stereotype_concepts/target_group_ref")

    plan.add_concept("individuals_classification")
    plan.make_reference("individuals_classification", "stereotype_concepts/individuals_classification_ref")

    plan.add_concept("individuals")

    # Set I/O and validate dependencies
    plan.configure_io(
        input_names=["target_group", "individuals_classification"],
        output_name="individuals"
    )

    # Configure inferences
    plan.add_inference(
        ["target_group"],
        "individuals_classification",
        "individuals",
        view=["target_group"]
    )

    statement_single_input = lambda statement: Reference(
        axes = ["statement"],
        shape = (1,),
        initial_value = statement,
    )

    statement_input = statement_single_input(
        "engineers are happy."
    )

    # Execute the full plan
    output_ref = plan.execute()

    # Display results
    print("\nFinal Inference Results:")
    print("Reference Tensor:", output_ref.tensor)
    print("Tensor Axes:", output_ref.axes)

    # Inspect agent memory state
    print("\nAgent Memory State:")
    print("Working Memory:", agent.working_memory)
    print("Persisted Memory:", json.dumps(json.load(open("../memory.json")), indent=2))
