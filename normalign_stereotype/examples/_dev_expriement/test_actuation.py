from normalign_stereotype.core._agent_frame import AgentFrame
from normalign_stereotype.core._llm_tools import ConfiguredLLM, JsonBulletLLM, JsonStructuredLLM
from typing import Optional, List, Union, Dict, Set, Tuple, Any
from normalign_stereotype.core._objects import Plan
from pathlib import Path



if __name__ == "__main__":
    # Create agent with memory location and debug mode
    llm_model_name = "qwen-turbo-latest"

    agent = AgentFrame(
        body={
            'memory_location': str(Path.cwd() / "memory.json"),
            'llm': ConfiguredLLM(model_name=llm_model_name),
            "bullet_llm": JsonBulletLLM(model_name=llm_model_name),
            "structured_llm": JsonStructuredLLM(model_name=llm_model_name),
        },
        debug=True  # Ensure debug mode is enabled
    )

    # Create a new plan with debug mode
    plan = Plan(debug=True)  # Ensure debug mode is enabled

    # Add concepts to the plan
    figurative_concept = plan.add_concept(
        name="figurative_language_element",
        context="",
        type="{}"
    )
    
    entity_concept = plan.add_concept(
        name="specific_entity",
        context="",
        type="{}"
    )
    
    theme_concept = plan.add_concept(
        name="abstract_theme",
        context="",
        type="{}"
    )
    
    actuation_concept = plan.add_concept(
        name="{1}_maps_{2}_onto_{3}_directly",
        context="",
        type="<>"
    )
    
    concept_to_infer = plan.add_concept(
        name="figurative_language_element_maps_specific_entity_onto_abstract_theme_directly",
        context="",
        type="^"
    )

    # Add inference to the plan
    plan.add_inference(
        concept_to_infer=concept_to_infer,
        perception_concepts=[figurative_concept, entity_concept, theme_concept],
        actuation_concept=actuation_concept,
        view=["figurative_language_element"]
    )

    # Process actuation concept as a constant before execution
    constant_data = {
        "{1}_maps_{2}_onto_{3}_directly": """[{"Explanation": "The relation '{1}_maps_{2}_onto_{3}_directly' is true if and only if the following conditions are met: {1} must be a figurative language element that establishes a direct symbolic connection between {2}, a specific and tangible entity, and {3}, an abstract and complex theme, such that the meaning of {2} symbolically represents or conveys {3} without any intervening concepts or ambiguities obstructing the transfer of symbolic meaning. Additionally, the mapping must occur explicitly within a context where both {2} and {3} are clearly present and the symbolic relationship is unambiguous.", "Summary_Key": "{1}_maps_{2}_onto_{3}_directly"}]"""
    }
    plan.direct_reference_to_concept(agent, input_mode="raw", input_data=constant_data)

    # Set input and output concepts (excluding constants)
    plan.set_input_output_concepts(
        input_names=["figurative_language_element", "specific_entity", "abstract_theme"],
        output_name="figurative_language_element_maps_specific_entity_onto_abstract_theme_directly"
    )

    # Define input data for execution (excluding constants)
    input_data = {
        "figurative_language_element": """[{"Explanation": "The pen trembled in the hand of the diplomat says something more about the diplomat nervousness and uncertainty, which makes the sentence more figurative.", "Summary_Key": "'that the pen trembled in the hand of the diplomat'"}]""",
        "specific_entity": """[{"Explanation": "The trembling pen shows the nervousness and uncertainty of diplomat, and it is a specific entity.", "Summary_Key": "Trembling Pen"}]""",
        "abstract_theme": """[{"Explanation": "As diplomat mentioned represents diplomatic peaceful relations, the trembling of pen leads to the nervousness and uncertainty of the diplomat then further leads to the abstract theme of fragility of peace.", "Summary_Key": "Fragility of Peace"}]"""
    }

    # Execute the plan
    result = plan.execute(agent, input_mode="raw", input_data=input_data)
    print("Plan execution result:", result.tensor)
    print("Result axes:", result.axes)