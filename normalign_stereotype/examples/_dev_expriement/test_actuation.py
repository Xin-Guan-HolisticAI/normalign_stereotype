from string import Template
import copy
import re
from normalign_stereotype.core._modified_llm import ConfiguredLLM, StructuredLLM, BulletLLM


def _safe_eval(x):
    """Safely evaluate a string or return appropriate list format.
    
    Handles three cases:
    1. String representation of list: "[a, b, c]" -> [a, b, c]
    2. Single string value: "a" -> ["a"]
    3. List: [a, b, c] -> [a, b, c]
    """
    if isinstance(x, str):
        try:
            result = eval(x)
            if not isinstance(result, list):
                return [x]
            return result
        except:
            return [x]
    return x if isinstance(x, list) else [x]


def _replace_placeholders_with_values(template, values):
    """Replace placeholders in template with values from list.
    
    Args:
        template: String with placeholders like {1}, {2}, etc.
        values: List of values to replace placeholders with
        
    Returns:
        String with placeholders replaced by values
    """
    result = template
    for i, value in enumerate(values):
        placeholder = f'{{{i+1}}}'
        result = result.replace(placeholder, str(value).strip())
    return result.replace('_', ' ')


def _format_bullet_points(concept_names, instance_names, context_values):
    """Format concept names, instance names, and context values into bullet points.
    
    Args:
        concept_names: List of concept names
        instance_names: List of instance names
        context_values: List of context values
        
    Returns:
        String of formatted bullet points
    """
    return ''.join(f" - {i+1}: the instance of {c} is {n}. (context: {v})\\n" 
                  for i, (c, n, v) in enumerate(zip(concept_names, instance_names, context_values)))


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

        classification_prompt = Template("""

Your task is to find instances of "$actu_n" from a specific text about an instance of "$input_concept_name".
    
What "$actu_n" means: "$actu_v"

**Find from "$perc_cn": "$perc_n"**

(context for "$perc_n": "$perc_v")

Your output should start with some context, reasonings and explanations of the existence of the instance. Your summary key should be an instance of "$actu_n".

    """)
        
        actuation_config = {
            "mode": "llm_prompt_two_replacement",
            "llm": "structured_llm",
            "prompt_template": classification_prompt,
            "template_variable_definition_dict": default_variable_definition_dict,
        }
        
    elif concept_type == "<>":

        judgement_prompt = Template("""

Your task is to judge if "$actu_n_with_perc_n" is true or false.

What each of the component in "$actu_n_with_perc_n" refers to: 
$perc_cn_n_v_bullets
                            
**Truth conditions to judge if it is true or false that "$actu_n_with_perc_n":** 
    "$actu_v_with_perc_n"

When judging **quote** the specific part of the Truth conditions you mentioned to make the judgement in your output, this is to make sure you are not cheating and the answer is intelligible without the Truth conditions.
                            
Now, judge if "$actu_n_with_perc_n" is true or false based **strictly** on the above Truth conditions, and quote the specific part of the Truth conditions you mentioned to make the judgement in your output.

Your output should start with some context, reasoning and explanations of the existence of the instance. Your summary key should be an "TRUE" or "FALSE" (or "N/A" if not applicable).

    """)

        actuation_config = {
            "mode": "llm_prompt_two_replacement",
            "llm": "bullet_llm",
            "prompt_template": judgement_prompt,
            "template_variable_definition_dict": default_variable_definition_dict
        }

    return perception_config, actuation_config




def _clean_parentheses( text):
    # Remove parentheses content then clean up spaces
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
    base_values_dict: dict
) -> str:
    """
    Substitutes only variables that can be resolved with available locals.
    Leaves unresolved variables in the template unchanged.
    
    Args:
        prompt_template: Template string or Template object with placeholders
        template_variable_definition_dict: Dict mapping variables to code snippets that generate values
        base_values_dict: Variables available for code execution
        
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
        exec_env = copy.deepcopy(base_values_dict)
        
        try:
            # Execute code in isolated environment
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

    # memory_location = self.body.get("memory_location")
    memory_location = "memory.json"
    memory = eval(open(memory_location).read())

    base_values_dict = {}

    to_actuate_value = _clean_parentheses(memory.get(key_build(to_actuate_name), to_actuate_name))
    base_values_dict["actu_v"] = to_actuate_value
    base_values_dict["actu_n"] = to_actuate_name
    base_values_dict["actu_cn"] = to_actuate_concept_name if to_actuate_concept_name is not None else "actu_cn"
    base_values_dict["perc_cn"] = perception_concept_name if perception_concept_name is not None else "perc_cn"

    actuated_prompt = _prompt_template_dynamic_substitution(prompt_template, variable_definitions, base_values_dict)

    def actuated_func(input_perception):
        
        perception_name = _clean_parentheses(str(input_perception[0]))
        perception_value = str(input_perception[1])
        base_values_dict["perc_n"] = perception_name
        base_values_dict["perc_v"] = perception_value

        acutated_prompt_with_perception = _prompt_template_dynamic_substitution(actuated_prompt, variable_definitions, base_values_dict)
        print("         acutated_prompt_with_perception:  ", repr(acutated_prompt_with_perception))
        return eval(actuated_llm.invoke(acutated_prompt_with_perception))

    return actuated_func


if __name__ == "__main__":




    back_up_concept_meaning_response = "The relation \"{1}_maps_{2}_onto_{3}_directly\" is true if and only if the following conditions are met: {1} must be a figurative language element that establishes a direct symbolic connection between {2}, a specific and tangible entity, and {3}, an abstract and complex theme, such that the meaning of {2} symbolically represents or conveys {3} without any intervening concepts or ambiguities obstructing the transfer of symbolic meaning. Additionally, the mapping must occur explicitly within a context where both {2} and {3} are clearly present and the symbolic relationship is unambiguous."




    # Create a template with placeholders
    template_str = """
Actuation Details:
- Name: $actu_n ($actu_cn)
- Value: $actu_v
- Formatted Value: $formatted_actu_v
- Combined Info: $combined_info

Perception Details:
- Name: $perc_n ($perc_cn)
- Value: $perc_v
- Formatted Perception: $formatted_perception
- Analysis: $perception_analysis

Summary:
$summary
"""
    
    # Define how to generate values for each placeholder
    variable_definitions = {
        "actu_n": "actu_n = actu_n",  # Direct use of base value
        "actu_cn": "actu_cn = actu_cn",  # Direct use of base value
        "actu_v": "actu_v = actu_v",  # Direct use of base value
        "formatted_actu_v": "formatted_actu_v = f'Value: {actu_v} (from {actu_n})'",  # Using base values directly
        "combined_info": "combined_info = f'{actu_n} ({actu_cn}): {actu_v}'",  # Combining multiple base values
        
        "perc_n": "perc_n = perc_n",  # Direct use of base value
        "perc_cn": "perc_cn = perc_cn",  # Direct use of base value
        "perc_v": "perc_v = perc_v",  # Direct use of base value
        "formatted_perception": "formatted_perception = f'Observed {perc_n} ({perc_cn}) = {perc_v}'",  # Using base values directly
        "perception_analysis": "perception_analysis = f'When {actu_n} is {actu_v}, we observe {perc_n} as {perc_v}'",  # Combining actuation and perception values
        
        "summary": "summary = f'Actuation {actu_n} with value {actu_v} resulted in perception {perc_n} = {perc_v}'"  # Comprehensive summary using all values
    }
    
    # Base values that might be needed for calculations
    base_values = {
        "actu_n": "test_actuation",
        "actu_cn": "test_concept_actuation",
        "actu_v": "test_value",
        "perc_n": "test_perception",
        "perc_cn": "perception_concept",
        "perc_v": "perception_value"
    }
    
    # Test the function with string template
    result = _prompt_template_dynamic_substitution(
        template_str,
        variable_definitions,
        base_values
    )
    
    print("Template:", template_str)
    print("Result:", result)
    
    # Test _actuation_llm_prompt_two_replacement
    print("\nTesting _actuation_llm_prompt_two_replacement:")
    
    # Create actuated function
    actuated_func = _actuation_llm_prompt_two_replacement(
        to_actuate_name="test_actuation",
        prompt_template=template_str,
        variable_definitions=variable_definitions,
        key_build=lambda x:_key_memory(x, "test_concept_actuation"),
        actuated_llm=BulletLLM(model_name="qwen-turbo-latest"),
        to_actuate_concept_name="test_concept_actuation",
        perception_concept_name="perception_concept"
    )
    
    # Test with a perception
    perception = ("test_perception", "perception_value")
    result = actuated_func(perception)
    print("Actuation result:", result)
    
