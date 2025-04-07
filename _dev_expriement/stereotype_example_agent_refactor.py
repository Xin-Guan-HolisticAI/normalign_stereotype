from _modified_llm import ConfiguredLLM, BulletLLM, StructuredLLM
from _reference import Reference, cross_action, cross_product, element_action
from _concept import Concept
from _agent import Agent

import json
import os


def _chunk_list_by_value(lst, start_value=None, end_value=None):
    """
    Returns the sublist between the first occurrence of start_value (exclusive) and
    the first occurrence of end_value (exclusive) that comes after it.
    If either value is None, the chunk extends to the beginning or end of the list.

    Args:
        lst (list): The input list
        start_value: The value marking the start of the chunk (exclusive). If None, starts from beginning.
        end_value: The value marking the end of the chunk (exclusive). If None, goes until end.

    Returns:
        list: A new list containing elements between start_value and end_value.
              Returns an empty list if:
              - start_value is specified but not found
              - end_value is specified but not found after start_value
              - input isn't a list
    """
    if not isinstance(lst, list):
        return []

    try:
        start_index = 0
        if start_value is not None:
            start_index = lst.index(start_value) + 1  # +1 to exclude start_value itself

        end_index = len(lst)
        if end_value is not None:
            # Find end_value after the start_index
            end_index = lst.index(end_value, start_index)

        return lst[start_index:end_index]
    except ValueError:
        # Either start_value or end_value not found in the expected positions
        return []

if __name__ == '__main__':

    # Your existing LLM instances
    structured_llm = StructuredLLM()  # For structured outputs
    llm = ConfiguredLLM()  # Primary LLM
    bullet_llm = BulletLLM()  # For bullet-point processing

    if not os.path.exists("../memory.json"):
        with open("../memory.json", "w") as f:
            json.dump({}, f)  # Initialize empty JSON

    # Create the 'body' dictionary
    body = {
        "llm": llm,  # Primary LLM (required)
        "structured_llm": structured_llm,  # Optional (for structured tasks)
        "bullet_llm": bullet_llm,  # Optional (for bullet-point tasks)
        "memory_location": "memory.json",  # Path to memory file (required)
    }

    #create agent
    agent = Agent(body)

    #code for concept creation
    concept_creation_code = """globals()[f"{concept_name}_ref_tensor"] = eval(open(f"stereotype_concepts/{concept_name}_ref", encoding="utf-8").read())
globals()[f"{concept_name}_ref"] = Reference(
    axes=[concept_name],
    shape=(len(globals()[f"{concept_name}_ref_tensor"]),),
    initial_value=0,
)
globals()[f"{concept_name}_ref"].tensor = globals()[f"{concept_name}_ref_tensor"]
print(globals()[f"{concept_name}_ref"].tensor)

# Initialize target group concept
globals()[f"{concept_name}_concept"] = Concept(
    name=concept_name,
    context=concept_name,
    reference=globals()[f"{concept_name}_ref"]
    )
"""
    #initiate the target group reference and concept
    concept_name = "target_group"
    exec_globals = globals().copy()
    exec(concept_creation_code, exec_globals)
    globals()[f"{concept_name}_concept"]= exec_globals[f"{concept_name}_concept"]
    print(globals()[f"{concept_name}_concept"])

    perception_config = {
        "mode": "memory_retrieval"
    }

    globals()[f"{concept_name}_ref_cog"] = agent.cognition(globals()[f"{concept_name}_concept"], perception=perception_config)

    concept_name = "individuals_classification"
    exec_globals = globals().copy()
    exec(concept_creation_code, exec_globals)
    globals()[f"{concept_name}_concept"] = exec_globals[f"{concept_name}_concept"]
    print(globals()[f"{concept_name}_concept"])

    actuation_config = {
        "mode": "classification",
        "actuated_llm": "structured_llm",
        "prompt_template_path": "basic_template/classification-d",
        "place_holders" : {
            "meta_input_name_holder": "{meta_input_name}",
            "meta_input_value_holder": "{meta_input_value}",
            "input_key_holder": "{input_name}",
            "input_value_holder": "{input_value}",
        },
    }
    globals()[f"{concept_name}_ref_cog"] = agent.cognition(globals()[f"{concept_name}_concept"], actuation=actuation_config)

    print(agent.working_memory)

    """===== SETTING COMPLETES ====="""

    """===== perception start ====="""
    concept_name = "target_group"
    globals()[f"{concept_name}_concept"].reference = globals()[f"{concept_name}_ref_cog"]
    print(globals()[f"{concept_name}_concept"].reference.tensor)

    globals()[f"{concept_name}_ref_per"] = agent.perception(globals()[f"{concept_name}_concept"])
    print("PERCEPTION TENSOR:")
    print(globals()[f"{concept_name}_ref_per"].tensor)
    B = globals()[f"{concept_name}_ref_per"]

    """===== actuation start ====="""
    concept_name = "individuals_classification"
    globals()[f"{concept_name}_concept"].reference = globals()[f"{concept_name}_ref_cog"]
    print(globals()[f"{concept_name}_concept"].reference.tensor)

    globals()[f"{concept_name}_ref_act"] = agent.actuation(globals()[f"{concept_name}_concept"])
    print(globals()[f"{concept_name}_ref_act"].tensor)
    A = globals()[f"{concept_name}_ref_act"]

    """===== inference main execution start ====="""
    concept_name = "individuals"
    globals()[f"{concept_name}_ref_raw"] = cross_action(A, B, concept_name)
    print(globals()[f"{concept_name}_ref_raw"].tensor)
    print(globals()[f"{concept_name}_ref_raw"].axes)

    """===== view change start ====="""
    axes = globals()[f"{concept_name}_ref_raw"].axes
    new_view = _chunk_list_by_value(axes, start_value=A.axes[-1])

    globals()[f"{concept_name}_ref_vwc"] = globals()[f"{concept_name}_ref_raw"].slice(*new_view)
    print(globals()[f"{concept_name}_ref_vwc"].tensor)
    print(globals()[f"{concept_name}_ref_vwc"].axes)

    #view_change is not simple slicing - there will also require re-configuration of name/perception/name/actuation for the list etc. - probably need to include this in the agent if it uses llm.

    """===== inference main - creating concept -  start ====="""
    globals()[f"{concept_name}_concept"] = Concept(
        name=concept_name,
        context=concept_name,
        reference=globals()[f"{concept_name}_ref_vwc"]
    )

    """===== cognition configuration generation start ====="""

    perception_config = {
        "mode": "memory_retrieval"
    }
    actuation_config = {
        "mode": "pos",
        "actuated_llm": "bullet_llm",
        "meta_llm": "llm",
        "prompt_template_path": "pos_template/noun",
        "place_holders" : {
            "meta_input_name_holder": "{meta_input_name}",
            "meta_input_value_holder": "{meta_input_value}",
            "input_key_holder": "{verb/proposition}",
            "input_value_holder": "{input_value}",
        },
    }

    globals()[f"{concept_name}_ref_cog"] = agent.cognition(
        globals()[f"{concept_name}_concept"],
        actuation=actuation_config,
        perception=perception_config,
    )

    globals()[f"{concept_name}_concept"].reference = globals()[f"{concept_name}_ref_cog"]

    """===== inference completes ====="""
    print(globals()[f"{concept_name}_concept"].reference.tensor)
    print(globals()[f"{concept_name}_concept"].reference.axes)
    print(agent.working_memory)
