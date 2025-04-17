from string import Template
import copy
import re
import ast
import json
from normalign_stereotype.core._modified_llm import ConfiguredLLM, StructuredLLM, BulletLLM, JsonStructuredLLM, JsonBulletLLM
from normalign_stereotype.core._agent import Agent
from normalign_stereotype.core._concept import Concept
from normalign_stereotype.core._reference import Reference, element_action, cross_action, cross_product
import logging

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

def _cognition_memory_bullet(bullet, concept_name, memory_location):
    value, name = bullet.rsplit(':', 1)
    _update_memory(name.strip(), value.strip(), concept_name, memory_location)
    return name

def _update_memory(name, value, concept_name, memory_location):
    """Persist data to JSON file using JSON Lines format (one JSON object per line)"""

    _key_memory_concept = lambda x:_key_memory(x, concept_name)

    with open(memory_location, 'r+') as f:
        data = json.load(f)  # Reads and moves pointer to EOF
        data[_key_memory_concept(name)] = value  # Modify data
        f.seek(0)  # Move pointer back to start
        json.dump(data, f)  # Write new data (overwrites from position 0)
        f.truncate()

def _cognition_memory_json_bullet(json_bullet, concept_name, memory_location):
    """Process JSON bullet points and update memory"""
    try:
        # # Clean up the JSON string by removing any escaped quotes and ensuring proper format
        # cleaned_json = json_bullet.replace('\\"', '"').replace("'", '"')
        
        # Parse JSON bullet points
        bullets = json.loads(json_bullet)
        if not isinstance(bullets, list) or len(bullets) == 0:
            raise ValueError("Invalid JSON format: must be a non-empty list")
            
        # Get the first bullet point
        bullet = bullets[0]
        if not isinstance(bullet, dict):
            raise ValueError("Invalid bullet format: must be a dictionary")
            
        name = bullet.get("Summary_Key", "")
        value = bullet.get("Explanation", "")
        
        if not name or not value:
            raise ValueError("Missing required fields: Summary_Key or Explanation")

        # Clean up the name by removing parentheses and extra spaces
        name = re.sub(r'[()]', '', name).strip()
        
        # Update memory
        _update_memory(name, value.strip(), concept_name, memory_location)
        return name
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {str(e)}")
        logging.error(f"JSON string: {json_bullet}")
        return None
    except Exception as e:
        logging.error(f"Error processing JSON bullet: {str(e)}")
        logging.error(f"JSON string: {json_bullet}")
        return None


def _combine_pre_perception_concepts(pre_perception_concepts, agent: Agent):
    #use cross-product to make the only perception concept for processing
    the_pre_perception_concept_name = (
        str([pc.comprehension["name"] for pc in pre_perception_concepts])
        if len(pre_perception_concepts) > 1
        else pre_perception_concepts[0].comprehension["name"]
    )
    the_pre_perception_reference = (
        cross_product(
            [pc.reference for pc in pre_perception_concepts]
        )
        if len(pre_perception_concepts) > 1
        else pre_perception_concepts[0].reference
    )

    the_pre_perception_concept_type = "[]"

    agent.working_memory['perception'][the_pre_perception_concept_name], _ = \
        get_default_working_config(the_pre_perception_concept_type)

    return Concept(
        name = the_pre_perception_concept_name,
        context = "",
        type = the_pre_perception_concept_type,
        reference = the_pre_perception_reference,
    )




if __name__ == "__main__":

    class AgentHere(Agent):

        def __init__(self, body):
            super().__init__(body)

        def cognition(self, concept, mode_of_remember = "memory_bullet", perception_working_config = None, actuation_working_config = None, **kwargs):
            """Process values into names and store"""

            if not isinstance(concept, Concept):
                raise ValueError("Perception requires Concept instance")

            raw_reference = concept.reference
            concept_name = concept.comprehension.get("name")
            concept_context = concept.comprehension.get("context")
            concept_type = concept.comprehension.get("type")

            default_perception_working_config, default_actuation_working_config = get_default_working_config(concept_type)
            self.working_memory['perception'][concept_name] = perception_working_config or default_perception_working_config
            self.working_memory['actuation'][concept_name] = actuation_working_config or default_actuation_working_config


            if mode_of_remember == "memory_bullet":
                _cognition_memory_bullet_element = lambda bullet: _cognition_memory_bullet(
                    bullet,
                    concept_name,
                    self.body['memory_location']
                )
                return element_action(_cognition_memory_bullet_element, [raw_reference])
            elif mode_of_remember == "memory_json_bullet":
                _cognition_memory_json_bullet_element = lambda json_bullet: _cognition_memory_json_bullet(
                    json_bullet,
                    concept_name,
                    self.body['memory_location']
                )
                return element_action(_cognition_memory_json_bullet_element, [raw_reference])

            raise ValueError(f"Unknown cognition mode: {mode_of_remember}")
        
        def perception(self, concept):
            """Retrieve values through different perception modes"""

            if not isinstance(concept, Concept):
                raise ValueError("Perception requires Concept instance")

            reference = concept.reference

            concept_name_may_list_str = concept.comprehension.get("name")
            concept_name_may_list = (
                eval(concept_name_may_list_str)
                if (concept_name_may_list_str.startswith("[")
                    and concept_name_may_list_str.endswith("]"))
                else concept_name_may_list_str
            )

            perception_configuration = self.working_memory['perception']
            concept_configuration = perception_configuration.get(str(concept_name_may_list))
            _key_memory_concept = lambda x:self._key_memory(x, concept_name_may_list)

            mode = concept_configuration.get("mode")

            if mode == 'memory_retrieval':
                _memory_retrieval_perception = lambda name_may_list:(
                    self._perception_memory_retrieval(
                        _key_memory_concept(name_may_list)
                    )
                )

                return element_action(_memory_retrieval_perception, [reference])
            
        def _perception_memory_retrieval(self, name_may_list):
            """File-based value retrieval from JSONL storage"""
            with open(self.body['memory_location'], 'r', encoding='utf-8') as f:
                memory = eval(f.read())
                if isinstance(name_may_list,list):
                    name_list = name_may_list
                    value_list = []
                    for name in name_list:
                        value = memory.get(name)
                        value_list.append(value)
                    return [name_list, value_list]
                else:
                    name = name_may_list
                    value = memory.get(name)
                    return [name, value]


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
            # "bullet_llm": BulletLLM(model_name="qwen-turbo-latest"),
            "bullet_llm": JsonBulletLLM(model_name="qwen-turbo-latest"),
            # "structured_llm": StructuredLLM(model_name="qwen-turbo-latest"),
            "structured_llm": JsonStructuredLLM(model_name="qwen-turbo-latest"),
            "memory_location": "memory.json"
        }
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
        processed_reference = agent.cognition(concept, mode_of_remember="memory_json_bullet")
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
    actuated_reference = agent.cognition(raw_actuation_concept, mode_of_remember="memory_json_bullet")
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
    
