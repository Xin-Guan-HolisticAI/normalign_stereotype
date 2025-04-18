from normalign_stereotype.core._agent import (
    AgentFrame,
    _combine_pre_perception_concepts
)
from normalign_stereotype.core._concept import Concept
from normalign_stereotype.core._reference import Reference, element_action, cross_action
from normalign_stereotype.core._modified_llm import BulletLLM, JsonBulletLLM, JsonStructuredLLM, ConfiguredLLM

import logging

if __name__ == "__main__":
    # Create agent with memory location
    llm_model_name = "qwen-turbo-latest"

    agent = AgentFrame(
        body={
            'memory_location': 'memory.json',
            'llm': ConfiguredLLM(model_name=llm_model_name),
            "bullet_llm": JsonBulletLLM(model_name=llm_model_name),
            "structured_llm": JsonStructuredLLM(model_name=llm_model_name),
                # Replace with actual LLM instance
        },
        debug=True
    )

    # Define the perception concepts
    figurative_language_element = """'The pen trembled in the hand of the diplomat.'"""
    specific_entity = "pen trembled"
    abstract_theme = "Fragility of Peace"

    # Create raw perception concepts
    raw_perception_concepts: list[Concept] = []
    
    # Create figurative language element concept
    figurative_concept = Concept(
        name="figurative_language_element",
        context="",
        type="[]"
    )
    figurative_reference = Reference(
        axes=["figurative_language_element"],
        shape=(1,),
        initial_value="""[{"Explanation": "The pen trembled in the hand of the diplomat says something more about the diplomat nervousness and uncertainty, which makes the sentence more figurative.", "Summary_Key": "'that the pen trembled in the hand of the diplomat'"}]"""
    )
    figurative_concept.reference = figurative_reference
    raw_perception_concepts.append(figurative_concept)

    # Create specific entity concept
    entity_concept = Concept(
        name="specific_entity",
        context="",
        type="[]"
    )
    entity_reference = Reference(
        axes=["specific_entity"],
        shape=(1,),
        initial_value='[{"Explanation": "The trembling pen shows the nervousness and uncertainty of diplomat, and it is a specific entity.", "Summary_Key": "Trembling Pen"}]'
    )
    entity_concept.reference = entity_reference
    raw_perception_concepts.append(entity_concept)

    # Create abstract theme concept
    theme_concept = Concept(
        name="abstract_theme",
        context="",
        type="[]"
    )
    theme_reference = Reference(
        axes=["abstract_theme"],
        shape=(1,),
        initial_value='[{"Explanation": "As diplomat mentioned represents diplomatic peaceful relations, the trembling of pen leads to the nervousness and uncertainty of the diplomat then further leads to the abstract theme of fragility of peace.", "Summary_Key": "Fragility of Peace"}]'
    )
    theme_concept.reference = theme_reference
    raw_perception_concepts.append(theme_concept)

    # Process each raw perception concept through cognition
    pre_perception_concepts: list[Concept] = []
    for concept in raw_perception_concepts:
        processed_reference = agent.cognition(concept)
        concept.reference = processed_reference
        pre_perception_concepts.append(concept)

    # Combine the raw perception concepts into a single pre-perception concept
    the_pre_perception_concept = _combine_pre_perception_concepts(pre_perception_concepts, agent)

    # Create actuation concept
    raw_actuation_concept = Concept(
        name="{1}_maps_{2}_onto_{3}_directly",
        context="",
        type="<>"
    )
    raw_actuation_reference = Reference(
        axes=[raw_actuation_concept.comprehension["name"]],
        shape=(1,),
        initial_value='[{"Explanation": "The relation \\"{1}_maps_{2}_onto_{3}_directly\\" is true if and only if the following conditions are met: {1} must be a figurative language element that establishes a direct symbolic connection between {2}, a specific and tangible entity, and {3}, an abstract and complex theme, such that the meaning of {2} symbolically represents or conveys {3} without any intervening concepts or ambiguities obstructing the transfer of symbolic meaning. Additionally, the mapping must occur explicitly within a context where both {2} and {3} are clearly present and the symbolic relationship is unambiguous.", "Summary_Key": "{1}_maps_{2}_onto_{3}_directly"}]'
    )
    raw_actuation_concept.reference = raw_actuation_reference

    # Get working config for judgement type
    actuated_reference = agent.cognition(raw_actuation_concept)
    the_actuation_concept = raw_actuation_concept
    the_actuation_concept.reference = actuated_reference

    # Get actuated function from agent
    actuated_func_reference = agent.actuation(
        concept=the_actuation_concept,
        for_perception_concept_name=the_pre_perception_concept.comprehension["name"]
    )

    percived_value_reference = agent.perception(
        concept=the_pre_perception_concept,
    )

    # Test with a perception
    result = cross_action(
        A=actuated_func_reference,
        B=percived_value_reference, 
        new_axis_name="figurative_language_element_maps_specific_entity_onto_abstract_theme_directly"
    )
    print("Actuation result:", result.tensor)



#     # Create a template with placeholders
#     template_str = """
# Actuation Details:
# - Name: $actu_n ($actu_cn)
# - Value: $actu_v
# - Formatted Value: $formatted_actu_v
# - Combined Info: $combined_info

# Perception Details:
# - Name: $perc_n ($perc_cn)
# - Value: $perc_v
# - Formatted Perception: $formatted_perception
# - Analysis: $perception_analysis

# Summary:
# $summary
# """
    
#     # Define how to generate values for each placeholder
#     variable_definitions = {
#         "actu_n": "actu_n = actu_n",  # Direct use of base value
#         "actu_cn": "actu_cn = actu_cn",  # Direct use of base value
#         "actu_v": "actu_v = actu_v",  # Direct use of base value
#         "formatted_actu_v": "formatted_actu_v = f'Value: {actu_v} (from {actu_n})'",  # Using base values directly
#         "combined_info": "combined_info = f'{actu_n} ({actu_cn}): {actu_v}'",  # Combining multiple base values
        
#         "perc_n": "perc_n = perc_n",  # Direct use of base value
#         "perc_cn": "perc_cn = perc_cn",  # Direct use of base value
#         "perc_v": "perc_v = perc_v",  # Direct use of base value
#         "formatted_perception": "formatted_perception = f'Observed {perc_n} ({perc_cn}) = {perc_v}'",  # Using base values directly
#         "perception_analysis": "perception_analysis = f'When {actu_n} is {actu_v}, we observe {perc_n} as {perc_v}'",  # Combining actuation and perception values
        
#         "summary": "summary = f'Actuation {actu_n} with value {actu_v} resulted in perception {perc_n} = {perc_v}'"  # Comprehensive summary using all values
#     }
    
#     # Base values that might be needed for calculations
#     base_values = {
#         "actu_n": "test_actuation",
#         "actu_cn": "test_concept_actuation",
#         "actu_v": "test_value",
#         "perc_n": "test_perception",
#         "perc_cn": "perception_concept",
#         "perc_v": "perception_value"
#     }
    
#     # Test the function with string template
#     result = _prompt_template_dynamic_substitution(
#         template_str,
#         variable_definitions,
#         base_values
#     )
    
#     print("Template:", template_str)
#     print("Result:", result)
    
#     # Test _actuation_llm_prompt_two_replacement
#     print("\nTesting _actuation_llm_prompt_two_replacement:")
    
#     # Create actuated function
#     actuated_func = _actuation_llm_prompt_two_replacement(
#         to_actuate_name="test_actuation",
#         prompt_template=template_str,
#         variable_definitions=variable_definitions,
#         key_build=lambda x:_key_memory(x, "test_concept_actuation"),
#         actuated_llm=BulletLLM(model_name="qwen-turbo-latest"),
#         to_actuate_concept_name="test_concept_actuation",
#         perception_concept_name="perception_concept"
#     )
    
#     # Test with a perception
#     perception = ("test_perception", "perception_value")
#     result = actuated_func(perception)
#     print("Actuation result:", result)
    
