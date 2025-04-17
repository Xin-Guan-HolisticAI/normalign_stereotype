from string import Template
import copy
import re
import ast
from normalign_stereotype.core._modified_llm import ConfiguredLLM, StructuredLLM, BulletLLM
from normalign_stereotype.core._agent import Agent
from normalign_stereotype.core._concept import Concept
from normalign_stereotype.core._reference import Reference, element_action, cross_action

def _safe_eval(s):
    """Safely evaluate a string representation of a list or other Python literal."""
    try:
        return ast.literal_eval(s)
    except:
        return s


def _format_bullet_points(cn_list, n_list, v_list):
    """Format lists into bullet points."""
    bullets = []
    for cn, n, v in zip(cn_list, n_list, v_list):
        bullets.append(f" - {cn}: {n} (context: {v})")
    return "\n".join(bullets)


def _replace_placeholders_with_values(template, values):
    """Replace numbered placeholders in a template with values."""
    for i, value in enumerate(values, 1):
        template = template.replace(f"{{{i}}}", str(value))
        template = template.replace("_", " ")
    return template


def _clean_parentheses(text):
    """Remove parentheses content then clean up spaces."""
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_default_working_config(concept_type):
    """Get default cognition configuration based on concept type."""
    perception_config = {
        "mode": "memory_retrieval"
    }

    default_variable_definition_dict = {
        "actu_n": "actu_n",
        "actu_cn": "actu_cn",
        "actu_v": "actu_v",
        "perc_n": "perc_n",
        "perc_cn": "perc_cn",
        "perc_v": "perc_v",
        "perc_cn_n_v_bullets": "cn_list, n_list, v_list = _safe_eval(perc_cn), _safe_eval(perc_n), _safe_eval(perc_v); perc_cn_n_v_bullets = _format_bullet_points(cn_list, n_list, v_list)",
        "actu_n_with_perc_n": "name_elements = _safe_eval(perc_n); actu_n_with_perc_n = _replace_placeholders_with_values(actu_n, name_elements)",
        "actu_v_with_perc_n": "name_elements = _safe_eval(perc_n); actu_v_with_perc_n = _replace_placeholders_with_values(actu_v, name_elements)"
    }

    if concept_type == "?":

        classification_prompt = Template("""Your task is to find instances of "$actu_n" from a specific text about an instance of "$input_concept_name".
    
What "$actu_n" means: "$actu_v"

**Find from "$perc_cn": "$perc_n"**

(context for "$perc_n": "$perc_v")

Your output should start with some context, reasonings and explanations of the existence of the instance. Your summary key should be an instance of "$actu_n".""")
        
        actuation_config = {
            "mode": "llm_prompt_two_replacement",
            "llm": "structured_llm",
            "prompt_template": classification_prompt,
            "template_variable_definition_dict": default_variable_definition_dict,
        }
        
    elif concept_type == "<>":

        judgement_prompt = Template("""Your task is to judge if "$actu_n_with_perc_n" is true or false.

What each of the component in "$actu_n_with_perc_n" refers to: 
$perc_cn_n_v_bullets
                            
**Truth conditions to judge if it is true or false that "$actu_n_with_perc_n":** 
    "$actu_v_with_perc_n"

When judging **quote** the specific part of the Truth conditions you mentioned to make the judgement in your output, this is to make sure you are not cheating and the answer is intelligible without the Truth conditions.
                            
Now, judge if "$actu_n_with_perc_n" is true or false based **strictly** on the above Truth conditions, and quote the specific part of the Truth conditions you mentioned to make the judgement in your output.

Your output should start with some context, reasoning and explanations of the existence of the instance. Your summary key should be an "TRUE" or "FALSE" (or "N/A" if not applicable). Format: '.... : TRUE/FALSE/N/A'""")

        actuation_config = {
            "mode": "llm_prompt_two_replacement",
            "llm": "bullet_llm",
            "prompt_template": judgement_prompt,
            "template_variable_definition_dict": default_variable_definition_dict
        }

    return perception_config, actuation_config


def _key_memory(name_may_list, concept_name_may_list):
    """Format name-concept pairs as key for searching in memory, handling both single values and lists"""

    # Both are lists - zip them together
    if isinstance(name_may_list, list) and isinstance(concept_name_may_list, list):
        if len(name_may_list) != len(concept_name_may_list):
            raise ValueError("name_may_list and concept_name_may_list must be same length when both are lists")
        return [f"{n} ({c})" for n, c in zip(name_may_list, concept_name_may_list)]

    # Name is list but concept is single - apply same concept to all names
    elif isinstance(name_may_list, list):
        return [f"{n} ({concept_name_may_list})" for n in name_may_list]

    # Concept is list but name is single - apply same name to all concepts
    elif isinstance(concept_name_may_list, list):
        return [f"{name_may_list} ({c})" for c in concept_name_may_list]

    # Both are single values - simple combination
    else:
        return f"{name_may_list} ({concept_name_may_list})"


def _prompt_template_dynamic_substitution(
    prompt_template: str | Template,
    template_variable_definition_dict: dict[str, str],
    base_values_dict: dict,
    helper_functions: dict
) -> str:
    """
    Substitutes only variables that can be resolved with available locals.
    Leaves unresolved variables in the template unchanged.
    
    Args:
        prompt_template: Template string or Template object with placeholders
        template_variable_definition_dict: Dict mapping variables to code snippets that generate values
        base_values_dict: Variables available for code execution
        helper_functions: Dictionary of helper functions to use in code execution
        
    Returns:
        Prompt with resolved substitutions, unresolved variables remain as placeholders
    """
    substitutions = {}
    
    # Convert string to Template if needed
    if isinstance(prompt_template, str):
        template = Template(prompt_template)
    else:
        template = prompt_template
    
    # Get all variables present in the template
    template_variables = template.get_identifiers()
    
    for var in template_variables:
        if var not in template_variable_definition_dict:
            continue  # No code snippet for this variable
            
        code = template_variable_definition_dict[var]
        
        # Create execution environment with both base values and helper functions
        exec_env = copy.deepcopy(base_values_dict)
        exec_env.update(helper_functions)
        
        try:
            # Execute code in isolated environment with helper functions available
            exec(code, {}, exec_env)
            
            # Use the variable name itself as the key
            if var in exec_env:
                substitutions[var] = exec_env[var]
        except Exception:
            # Skip substitution if any error occurs
            continue
            
    # Use safe_substitute to leave unresolved variables unchanged
    return template.safe_substitute(substitutions)


def _actuation_llm_prompt_two_replacement(to_actuate_name, prompt_template, variable_definitions, key_build,
                                            actuated_llm, to_actuate_concept_name = None, perception_concept_name = None):
    """Create an actuated function that processes perceptions using the given template and definitions.
    
    Args:
        to_actuate_name: Name of the concept to actuate
        prompt_template: Template for the prompt
        variable_definitions: Dictionary mapping template variables to code snippets
        key_build: Function to build memory keys
        actuated_llm: LLM to use for actuation
        to_actuate_concept_name: Optional name of the concept being actuated
        perception_concept_name: Optional name of the perception concept
    """
    # memory_location = self.body.get("memory_location")
    memory_location = "memory.json"
    memory = eval(open(memory_location).read())

    base_values_dict = {}

    to_actuate_value = _clean_parentheses(memory.get(key_build(to_actuate_name), to_actuate_name))
    
    base_values_dict["actu_v"] = to_actuate_value
    base_values_dict["actu_n"] = to_actuate_name
    base_values_dict["actu_cn"] = to_actuate_concept_name if to_actuate_concept_name is not None else "actu_cn"
    base_values_dict["perc_cn"] = perception_concept_name if perception_concept_name is not None else "perc_cn"

    # Define helper functions to pass to template substitution
    helper_functions = {
        '_safe_eval': _safe_eval,
        '_format_bullet_points': _format_bullet_points,
        '_replace_placeholders_with_values': _replace_placeholders_with_values,
        '_clean_parentheses': _clean_parentheses
    }

    actuated_prompt = _prompt_template_dynamic_substitution(
        prompt_template, 
        variable_definitions, 
        base_values_dict,
        helper_functions
    )

    def actuated_func(input_perception):
        perception_name = _clean_parentheses(str(input_perception[0]))
        perception_value = str(input_perception[1])
        
        base_values_dict["perc_n"] = perception_name
        base_values_dict["perc_v"] = perception_value

        acutated_prompt_with_perception = _prompt_template_dynamic_substitution(
            actuated_prompt, 
            variable_definitions, 
            base_values_dict,
            helper_functions
        )
        print("\n[DEBUG] Final prompt before LLM call:", acutated_prompt_with_perception)
        return eval(actuated_llm.invoke(acutated_prompt_with_perception))

    return actuated_func


if __name__ == "__main__":

    class AgentHere(Agent):

        def __init__(self, body):
            super().__init__(body)

        def actuation(self, concept, for_perception_concept_name = ''):
            """Create functions through named parameter resolution"""

            if not isinstance(concept, Concept):
                raise ValueError("Actuation requires Concept instance")

            reference = concept.reference
            concept_name = concept.comprehension.get("name","")
            concept_context = concept.comprehension.get("context","")
            concept_type = concept.comprehension.get("type","{}")

            _key_memory_concept = lambda x:self._key_memory(x, concept_name)

            actuation_working_configuration = self.working_memory['actuation'].get(concept_name)
            mode = actuation_working_configuration.get("mode")

            if mode == "llm_prompt_two_replacement":
                actuated_llm = self.body[actuation_working_configuration.get("llm")]
                prompt_template = actuation_working_configuration.get("prompt_template")
                variable_definitions = actuation_working_configuration.get("template_variable_definition_dict")

                _actuated_funcn = lambda actu_name: (
                    _actuation_llm_prompt_two_replacement(
                    to_actuate_name=actu_name,
                    prompt_template=prompt_template,
                    variable_definitions=variable_definitions,
                    key_build=_key_memory_concept,
                    actuated_llm=actuated_llm,
                    to_actuate_concept_name=concept_name,
                    perception_concept_name=for_perception_concept_name
                    )
                )

            return element_action(_actuated_funcn, [reference])


    # Create agent with memory location
    agent = AgentHere(
        body={
            "llm": ConfiguredLLM(model_name="qwen-turbo-latest"),
            "bullet_llm": BulletLLM(model_name="qwen-turbo-latest"),
            "structured_llm": StructuredLLM(model_name="qwen-turbo-latest"),
            "memory_location": "memory.json"
        }
    )

    # Define the perception concepts
    figurative_language_element = """'The pen trembled in the hand of the diplomat.'"""
    specific_entity = "pen trembled"
    abstract_theme = "Fragility of Peace"

    # Create perception values
    perc_n_list = [figurative_language_element, specific_entity, abstract_theme]
    perc_v_list = [figurative_language_element, specific_entity, abstract_theme]
    perc_cn_list = ["figurative language element", "specific entity", "abstract theme"]

    # Create perception concept
    perception_concept = Concept(
        name=str(perc_cn_list),
        context="",
        type="[]"
    )
    # Create actuation concept
    actuation_concept = Concept(
        name="{1}_maps_{2}_onto_{3}_directly",
        context="",
        type="<>"
    )

    # Get working config for judgement type
    perception_config, actuation_config = get_default_working_config("<>")

    # Configure agent's working memory
    agent.working_memory = {
        'perception': {
            perception_concept.comprehension["name"]: perception_config
        },
        'actuation': {
            actuation_concept.comprehension["name"]: actuation_config
        }
    }

    # Create perception reference with proper structure - list of (name, value) pairs
    perception_reference = Reference(
        axes=[perception_concept.comprehension["name"]],
        shape=(1,),
        initial_value=list((perc_n_list, perc_v_list))  # Create list of (name, value) 
    )
    perception_concept.reference = perception_reference

    # Create actuation reference
    actuation_reference = Reference(
        axes=[actuation_concept.comprehension["name"]],
        shape=(1,),
        initial_value=actuation_concept.comprehension['name']
    )
    actuation_concept.reference = actuation_reference

    # Get actuated function from agent
    actuated_func_reference = agent.actuation(
        concept=actuation_concept,
        for_perception_concept_name=perception_concept.comprehension["name"]
    )

    # Test with a perception
    result = cross_action(
        A=actuated_func_reference,
        B=perception_reference,
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
    
