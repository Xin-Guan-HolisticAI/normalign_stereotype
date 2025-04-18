from ._utils import _safe_eval, _get_default_working_config
from normalign_stereotype.core._concept import Concept
from normalign_stereotype.core._reference import cross_product

def _perception_memory_retrieval(name_may_list, index_dict, recollection, memory_location):
    """File-based value retrieval from JSONL storage with location awareness"""
    with open(memory_location, 'r', encoding='utf-8') as f:
        memory = eval(f.read())
        
        if isinstance(name_may_list,list):
            name_list = name_may_list
            value_list = []
            for name in name_list:
                value = recollection(memory, name, index_dict)
                value_list.append(value)
            return [name_list, value_list]
        else:
            name = name_may_list
            value = recollection(memory, name, index_dict)
            return [name, value] 

def _combine_pre_perception_concepts(pre_perception_concepts, agent):
    """Combine multiple perception concepts into a single concept for processing."""
    # Use cross-product to make the only perception concept for processing
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
        _get_default_working_config(the_pre_perception_concept_type)

    return Concept(
        name = the_pre_perception_concept_name,
        context = "",
        type = the_pre_perception_concept_type,
        reference = the_pre_perception_reference,
    ) 