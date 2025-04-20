import logging
from normalign_stereotype.core._agent_frame._cognition import _recollect_nested

# Configure logging
logger = logging.getLogger(__name__)

def _perception_memory_retrieval(name_may_list, concept_name_may_list, index_dict, recollection, memory_location, debug=False):
    """File-based value retrieval from JSONL storage with location awareness
    
    Args:
        name_may_list: Name or list of names to retrieve
        index_dict: Dictionary for indexing
        recollection: Function to perform the recollection
        memory_location: Path to the memory file
        debug: If True, enables debug logging
    """
    if debug:
        logger.debug(f"Starting memory retrieval for name_may_list: {name_may_list}")
        logger.debug(f"Memory location: {memory_location}")
        logger.debug(f"Index dictionary: {index_dict}")
    
    try:
        with open(memory_location, 'r', encoding='utf-8') as f:
            memory = eval(f.read())
            if debug:
                logger.debug(f"Successfully loaded memory from {memory_location}")
                logger.debug(f"Memory content type: {type(memory)}")
                logger.debug(f"Full memory content: {memory}")


            if isinstance(concept_name_may_list, str):
                concept_name_may_list = [concept_name_may_list]
            
            concept_name_list = concept_name_may_list

            if isinstance(name_may_list, str):
                name_may_list = [name_may_list]
            
            if isinstance(name_may_list, list):
                name_list = name_may_list
                if debug:
                    logger.debug(f"Processing list of names: {name_list}")
                value_list = _recollect_nested(memory, name_list, concept_name_list, index_dict, recollection, debug)
                if debug:
                    logger.debug(f"Retrieved values: {value_list}")
                return [name_list, value_list]
    except Exception as e:
        logger.error(f"Error during memory retrieval: {str(e)}", exc_info=True)
        raise 

