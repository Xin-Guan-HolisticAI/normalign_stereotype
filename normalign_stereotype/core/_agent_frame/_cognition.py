import json
import logging
import re
from normalign_stereotype.core._objects._concept import Concept
from normalign_stereotype.core._objects._reference import cross_product
from normalign_stereotype.core._agent_frame._utils import _get_default_working_config
from normalign_stereotype.core._objects._reference import Reference
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


def _recollect_nested(memory, name, concept_name_list, index_dict, recollection, debug=False):
    """Helper function to handle nested list recollection"""
    if debug:
        logger.debug(f"Recollecting nested for name: {name}")
        logger.debug(f"Memory content: {memory}")
        logger.debug(f"Index dict content: {index_dict}")
    
    if isinstance(name, list):
        return [_recollect_nested(memory, n, concept_name_list, index_dict, recollection, debug) for n in name]
    result = recollection(memory, name, concept_name_list, index_dict)
    if debug:
        logger.debug(f"Recollection result for {name}: {result}")
    return result


def _create_concept_reference(concept_name: str, explanation: str, summary_key: Optional[str] = None, mode_of_remember: str = "memory_bullet") -> Reference:
    """Create a reference for a concept with an explicit value.
    
    Args:
        concept: The name of the concept
        value: The explicit value to assign to the concept
        summary: Optional summary for the reference. If None, uses value
        mode_of_remember: The mode of remember to use for the reference

    Returns:
        A Reference object containing the concept reference with the specified value
    """
    # at least one of explanation or summary_key must be provided
    if (explanation is None or explanation == "") and (summary_key is None or summary_key == ""):
        raise ValueError("At least one of explanation or summary_key must be provided")
    
    if summary_key is None or summary_key == "":
        summary_key = explanation

    if mode_of_remember == "memory_bullet":
        return Reference(
            axes=[concept_name],
            shape=(1,),
            initial_value=f"[{explanation} : {summary_key}]"
        )
    elif mode_of_remember == "memory_json_bullet":
        return Reference(
            axes=[concept_name],
            shape=(1,),
            initial_value=f'{{"Explanation": "{explanation}", "Summary_Key": "{summary_key}"}}'
        )
    else:
        raise ValueError(f"Unknown mode of remember: {mode_of_remember}")




def _remember_in_concept_name_location_dict(name, value, concept_name, memory_location, index_dict=None):
    """Persist data to JSON file using concept_name|name|location format"""
    # Create a key with pipe separator and sorted location info
    if index_dict:
        # Sort indices for consistent key format
        sorted_indices = '::'.join(f"{k}_{v}" for k, v in sorted(index_dict.items()))
        key = f"{concept_name}|{name}|{sorted_indices}"
    else:
        key = f"{concept_name}|{name}"
    
    with open(memory_location, 'r+') as f:
        data = json.load(f)
        data[key] = value
        f.seek(0)
        json.dump(data, f)
        f.truncate()

def _recollect_by_concept_name_location_dict(memory, name, concept_name_list, indices, debug=True):
    """Retrieve value from memory using concept_name|name|location format
    
    Args:
        memory: Dictionary containing memory entries
        name: Name to search for
        indices: Dictionary of indices to match
        debug: If True, enables debug logging
    """
    if debug:
        logger.debug(f"Starting recollection for name: {name}")
        logger.debug(f"Concept name list: {concept_name_list}")
        logger.debug(f"Indices to match: {indices}")
        logger.debug(f"Memory keys: {list(memory.keys())}")
    
    # Create target indices set
    target_indices = {f"{k}_{v}" for k, v in indices.items()}
    if debug:
        logger.debug(f"Target indices set: {target_indices}")
    
    # Find matching keys and check their indices
    for key in memory:
            
        # Get indices part if it exists
        key_parts = [part for part in key.split('|') if part]  # Filter out empty parts
        if len(key_parts) != 3:
            if debug:
                logger.debug(f"Skipping key (parsing key parts mistake): {key}")
            continue
        
        # Check if the first part of the key is in the concept_name_list
        if key_parts[0] not in concept_name_list:
            if debug:
                logger.debug(f"Skipping key (concept name mismatch): {key}")
            continue

        # Check if the name matches the second part of the key
        if key_parts[1] != name:
            if debug:
                logger.debug(f"Skipping key (name mismatch): {key}")
            continue
            
        # Check if target indices are a subset of stored indices or vice versa
        stored_indices = set(part for part in key_parts[2].split('::') if part)  # Filter out empty parts
        if debug:
            logger.debug(f"Checking key: {key}")
            logger.debug(f"Stored indices: {stored_indices}")
            logger.debug(f"Target indices subset check: {target_indices.issubset(stored_indices)}")
            logger.debug(f"Sorted indices subset check: {stored_indices.issubset(target_indices)}")
            
        if target_indices.issubset(stored_indices) or stored_indices.issubset(target_indices):
            if debug:
                logger.debug(f"Found matching key: {key}")
                logger.debug(f"Returning value: {memory[key]}")
            return memory[key]
            
    if debug:
        logger.debug("No matching key found")
    return None

def _combine_pre_perception_concepts_by_two_lists(pre_perception_concepts, agent):
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

def _cognition_memory_bullet(bullet, concept_name, memory_location, index_dict, remember):
    value, name = bullet.rsplit(':', 1)
    remember(name.strip(), value.strip(), concept_name, memory_location, index_dict)
    return name

def _cognition_memory_json_bullet(json_bullet, concept_name, memory_location, index_dict, remember):
    """Process JSON bullet points and update memory with location awareness.
    Accepts either a JSON string or a Python object (dict or list)."""
    try:
        # Handle both JSON strings and Python objects
        if isinstance(json_bullet, str):
            bullets = json.loads(json_bullet)
        else:
            bullets = json_bullet
            
        # Handle both list and dict inputs
        if isinstance(bullets, dict):
            # If it's a single dict, use it directly
            bullet = bullets
        elif isinstance(bullets, list) and len(bullets) > 0:
            # If it's a list, take the first item
            bullet = bullets[0]
        else:
            raise ValueError("Invalid format: must be a dictionary or non-empty list")
            
        if not isinstance(bullet, dict):
            raise ValueError("Invalid bullet format: must be a dictionary")
            
        name = bullet.get("Summary_Key", "")
        value = bullet.get("Explanation", "")
        
        if not name or not value:
            raise ValueError("Missing required fields: Summary_Key or Explanation")

        # Clean up the name by removing parentheses and extra spaces
        name = re.sub(r'[()]', '', name).strip()
        
        # Update memory with index information
        remember(name, value.strip(), concept_name, memory_location, index_dict)
        return name
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {str(e)}")
        logging.error(f"JSON string: {json_bullet}")
        return None
    except Exception as e:
        logging.error(f"Error processing JSON bullet: {str(e)}")
        logging.error(f"Input: {json_bullet}")
        return None
