import json
import re
import logging
import ast
from string import Template
import copy

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

def _prompt_template_dynamic_substitution(
    prompt_template: str | Template,
    template_variable_definition_dict: dict[str, str],
    base_values_dict: dict,
    helper_functions: dict
) -> str:
    """
    Substitutes only variables that can be resolved with available locals.
    Leaves unresolved variables in the template unchanged.
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

def _get_default_working_config(concept_type):
    """Get default cognition configuration based on concept type."""
    perception_config = {
        "mode": "memory_retrieval"
    }
    actuation_config = {}

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

Your output should be a JSON object with Explanation and Summary_Key fields. The Explanation should contain your reasoning and the specific part of the Truth conditions you mentioned. The Summary_Key should be either "TRUE", "FALSE", or "N/A" (if not applicable).""")

        actuation_config = {
            "mode": "llm_prompt_two_replacement",
            "llm": "bullet_llm",
            "prompt_template": judgement_prompt,
            "template_variable_definition_dict": default_variable_definition_dict
        }

    return perception_config, actuation_config
