def _recollect_nested(memory, name, index_dict, recollection):
    """Helper function to handle nested list recollection"""
    if isinstance(name, list):
        return [_recollect_nested(memory, n, index_dict, recollection) for n in name]
    return recollection(memory, name, index_dict)

def _perception_memory_retrieval(name_may_list, index_dict, recollection, memory_location):
    """File-based value retrieval from JSONL storage with location awareness"""
    with open(memory_location, 'r', encoding='utf-8') as f:
        memory = eval(f.read())
        
        if isinstance(name_may_list, list):
            name_list = name_may_list
            value_list = _recollect_nested(memory, name_list, index_dict, recollection)
            return [name_list, value_list]
        else:
            name = name_may_list
            value = recollection(memory, name, index_dict)
            return [name, value] 

