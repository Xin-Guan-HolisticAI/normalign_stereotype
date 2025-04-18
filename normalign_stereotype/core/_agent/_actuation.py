from string import Template
from ._utils import (
    _clean_parentheses,
    _safe_eval,
    _format_bullet_points,
    _replace_placeholders_with_values,
    _prompt_template_dynamic_substitution
)

def _actuation_llm_prompt_two_replacement(to_actuate_name, prompt_template, variable_definitions,
                                            actuated_llm, to_actuate_concept_name = None, perception_concept_name = None, index_dict=None, recollection=None):
    """Create an actuated function that processes perceptions using the given template and definitions."""
    memory_location = "memory.json"
    memory = eval(open(memory_location).read())

    base_values_dict = {}

    # Get to_actuate_value with location awareness
    to_actuate_value = recollection(memory, to_actuate_name, index_dict)
    if to_actuate_value is None:
        to_actuate_value = to_actuate_name

    def actuated_func(input_perception):
        perception_name = _clean_parentheses(str(input_perception[0]))
        perception_value = str(input_perception[1])
        
        base_values_dict["actu_v"] = to_actuate_value
        base_values_dict["actu_n"] = to_actuate_name
        base_values_dict["actu_cn"] = to_actuate_concept_name if to_actuate_concept_name is not None else "actu_cn"
        base_values_dict["perc_cn"] = perception_concept_name if perception_concept_name is not None else "perc_cn"
        base_values_dict["perc_n"] = perception_name
        base_values_dict["perc_v"] = perception_value

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

        return eval(actuated_llm.invoke(actuated_prompt))

    return actuated_func 