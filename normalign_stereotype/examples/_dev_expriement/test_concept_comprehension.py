from normalign_stereotype.core._llm_tools import ConfiguredLLM, StructuredLLM, BulletLLM
from string import Template

object_concept_meaning_prompt = Template("""
    
Given the context: "$concept_context".

Explain what "$concept_name" means, but make sure your explanation is **independent of the context**.

Your output should be a one-to-several sentences for definition. Your summary key should be "$concept_name".

    """)
    
classification_prompt = Template("""

Your task is to find instances of "$meta_name" from a specific text about an instance of "$input_concept_name".
    
What "$meta_name" means: "$meta_value"

**Find from "$input_concept_name": "$input_name"**

(context for "$input_name": "$input_value")

Your output should start with some context, reasonings and explanations of the existence of the instance. Your summary key should be an instance of "$meta_name".

    """)


sentence_concept_meaning_prompt = Template("""
Context: "$concept_context"

By observing the syntatical structure of the clauses in the context, "$concept_name" is extracted from the context "$concept_context".                                                      

In the relation, "$concept_name" will take the instances of a concept by "$number_placeholder_list", where the corresponding concept of the instances are "$number_placeholder_list_context". 
                                           
"$concept_name" can JUDGE whether the relation is true or false (or not applicable) based on the truth conditions and the subustituted instances. 
                                            
Now, Find the truth conditions for the relation "$concept_name", but make sure your condition is **complete andintelligible without the context**.                                         

Your output should be a one-to-several sentences for truth conditions. Make emphasis on the important terms in the truth conditions. Your summary key should be "$concept_name".

    """)

judgement_prompt = Template("""

Your task is to judge if "$meta_name_replaced" is true or false.

What each of the component in "$meta_name_replaced" refers to: 
"$input_name_value_formated_bullet_points"
                            
**Truth conditions to judge if it is true or false that "$meta_name_replaced":** 
    "$meta_value_replaced"

When judging **quote** the specific part of the Truth conditions you mentioned to make the judgement in your output, this is to make sure you are not cheating and the answer is intelligible without the Truth conditions.
                            
Now, judge if "$meta_name_replaced" is true or false based **strictly** on the above Truth conditions, and quote the specific part of the Truth conditions you mentioned to make the judgement in your output.

Your output should start with some context, reasoning and explanations of the existence of the instance. Your summary key should be an "TRUE" or "FALSE" (or "N/A" if not applicable).

    """)


def test_sentence_concept_judgement(sentence_concept_meaning_prompt, judgement_prompt):
    # Initialize the LLM
    model_name = "qwen-turbo-latest"
    llm = ConfiguredLLM(model_name=model_name)
    structured_llm = StructuredLLM(model_name=model_name)
    bullet_llm = BulletLLM(model_name=model_name)
    
    # Get the concept meaning
    concept_name_for_concept_meaning_prompt = "{1}_maps_{2}_onto_{3}_directly"
    concept_context_for_concept_meaning_prompt = "From an extract, a metaphor is a figurative language element that directly maps a specific, tangible entity onto an abstract, complex theme if both components are present and the tangible entity transfers symbolic meaning to the abstract theme without **intermediaries** or **gaps**."
    number_placeholder_list = "{1}, {2}, {3}"
    number_placeholder_list_context = "{1}: figurative language element, {2}: specific, tangible entity {3}: abstract, complex theme"


    concept_meaning_prompt = sentence_concept_meaning_prompt.substitute(
        concept_name=concept_name_for_concept_meaning_prompt,
        concept_context=concept_context_for_concept_meaning_prompt,
        number_placeholder_list=number_placeholder_list,
        number_placeholder_list_context=number_placeholder_list_context
    )

    print("Concept meaning prompt:")
    print(concept_meaning_prompt)

    # Get the concept meaning
    print("Concept meaning response:")
    # concept_meaning_response = llm.invoke(concept_meaning_prompt)
    # print(concept_meaning_response)

    back_up_concept_meaning_response = "The relation \"{1}_maps_{2}_onto_{3}_directly\" is true if and only if the following conditions are met: {1} must be a figurative language element that establishes a direct symbolic connection between {2}, a specific and tangible entity, and {3}, an abstract and complex theme, such that the meaning of {2} symbolically represents or conveys {3} without any intervening concepts or ambiguities obstructing the transfer of symbolic meaning. Additionally, the mapping must occur explicitly within a context where both {2} and {3} are clearly present and the symbolic relationship is unambiguous."

    # Get the judgement
    meta_name_for_judgement_prompt = concept_name_for_concept_meaning_prompt

    figurative_language_element = """'The pen trembled in the hand of the diplomat.'"""
    specific_entity = "pen trembled"
    abstract_theme = "Fragility of Peace"


    input_name_list_for_judgement_prompt = [figurative_language_element, specific_entity, abstract_theme]

    # input_name_list_for_judgement_prompt = ["'Time is like a river that flows endlessly, carrying us forward whether we want it to or not.'", "river", "time"]
    # input_name_list_for_judgement_prompt = ["'light is hope'", "light", "hope"]

    meta_name_replaced = meta_name_for_judgement_prompt
    for i, name_element in enumerate(input_name_list_for_judgement_prompt):
        placeholder = f"{{{i+1}}}"  # Using 1-based indexing to match {1}, {2}, {3}
        meta_name_replaced = meta_name_replaced.replace(placeholder, name_element)
    meta_name_replaced = meta_name_replaced.replace("_", " ")  # Replace underscores with spaces

    # Get the meta value
    meta_value_for_judgement_prompt = back_up_concept_meaning_response
    meta_value_for_judgement_prompt_replaced = meta_value_for_judgement_prompt
    for i, name_element in enumerate(input_name_list_for_judgement_prompt):
        placeholder = f"{{{i+1}}}"  # Using 1-based indexing to match {1}, {2}, {3}
        meta_value_for_judgement_prompt_replaced = meta_value_for_judgement_prompt_replaced.replace(placeholder, name_element)
    meta_value_for_judgement_prompt_replaced = meta_value_for_judgement_prompt_replaced.replace("_", " ")  # Replace underscores with spaces

    # input_value_list_for_judgement_prompt = ["'Time is like a river that flows endlessly, carrying us forward whether we want it to or not.'", "river", "time"]
    # input_value_list_for_judgement_prompt = ["'light is hope'", "light", "hope"]
    input_value_list_for_judgement_prompt = [figurative_language_element, specific_entity, abstract_theme]


    input_concept_name_list_for_judgement_prompt = ["figurative language element", "specific entity", "abstract theme"]


    input_name_value_formated_bullet_points = ""
    for i, (concept_name, input_name, input_value) in enumerate(zip(input_concept_name_list_for_judgement_prompt, input_name_list_for_judgement_prompt, input_value_list_for_judgement_prompt)):
        input_name_value_formated_bullet_points += f" - {i+1}: the instance of {concept_name} is {input_name}. (context: {input_value})\n"


    judgement_prompt = judgement_prompt.substitute(
        input_name_value_formated_bullet_points=input_name_value_formated_bullet_points,
        meta_name_replaced=meta_name_replaced,
        meta_value_replaced=meta_value_for_judgement_prompt_replaced
    )

    print("Judgement prompt:")
    print(judgement_prompt)

    # Get the judgement
    response = structured_llm.invoke(judgement_prompt)
    # Print the response
    print("Response:")
    print(response)


def test_object_concept_comprehension(concept_meaning_prompt, classification_prompt):
    # Initialize the LLM
    # model_name = "qwen-turbo-latest"
    # model_name = "deepseek-r1-distill-qwen-7b"
    model_name = "qwen-plus"
    llm = ConfiguredLLM(model_name=model_name)
    structured_llm = StructuredLLM(model_name=model_name)
    bullet_llm = BulletLLM(model_name=model_name)
    
    # Get the concept meaning
    concept_name_for_concept_meaning_prompt = "specific_tangible_entity"
    concept_context_for_concept_meaning_prompt = "From an extract, a metaphor is a figurative language element that directly maps a specific, tangible entity onto an abstract, complex theme if both components are present and the tangible entity transfers symbolic meaning to the abstract theme without intermediaries."
    concept_meaning_prompt = concept_meaning_prompt.substitute(
        concept_name=concept_name_for_concept_meaning_prompt,
        concept_context=concept_context_for_concept_meaning_prompt
    )

    print("Concept meaning prompt:")
    print(concept_meaning_prompt)

    # Get the concept meaning
    print("Concept meaning response:")
    concept_meaning_response = llm.invoke(concept_meaning_prompt)
    print(concept_meaning_response)

    # Get the classification
    meta_name_for_classification_prompt = concept_name_for_concept_meaning_prompt
    meta_value_for_classification_prompt = concept_meaning_response
    input_concept_name_for_classification_prompt = "figurative_language_element"
    input_name_for_classification_prompt = "light is hope"
    input_value_for_classification_prompt = "light is hope is extracted from the context: The light is the hope of the world."
    classification_prompt = classification_prompt.substitute(
        meta_name=meta_name_for_classification_prompt,
        meta_value=meta_value_for_classification_prompt,
        input_concept_name=input_concept_name_for_classification_prompt,
        input_name=input_name_for_classification_prompt,
        input_value=input_value_for_classification_prompt
    )

    # Get the classification
    response = structured_llm.invoke(classification_prompt)
    # Print the response
    print("Response:")
    print(response)

if __name__ == "__main__":
    # test_object_concept_comprehension(object_concept_meaning_prompt, classification_prompt) 
    test_sentence_concept_judgement(sentence_concept_meaning_prompt, judgement_prompt)