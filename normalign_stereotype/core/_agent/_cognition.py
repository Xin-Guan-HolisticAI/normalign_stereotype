import json
import logging
import re
from normalign_stereotype.core._concept import Concept
from normalign_stereotype.core._reference import cross_product
from normalign_stereotype.core._agent._utils import _get_default_working_config

def _remember_in_concept_name_location_dict(name, value, concept_name, memory_location, index_dict=None):
    """Persist data to JSON file using concept_name|name|location format"""
    # Create a key with pipe separator and sorted location info
    if index_dict:
        # Sort indices for consistent key format
        sorted_indices = '_'.join(f"{k}::{v}" for k, v in sorted(index_dict.items()))
        key = f"{concept_name}|{name}|{sorted_indices}"
    else:
        key = f"{concept_name}|{name}"
    
    with open(memory_location, 'r+') as f:
        data = json.load(f)
        data[key] = value
        f.seek(0)
        json.dump(data, f)
        f.truncate()

def _recollect_by_concept_name_location_dict(memory, name, indices):
    """Retrieve value from memory using concept_name|name|location format"""
    # Create target indices set
    target_indices = {f"{k}::{v}" for k, v in indices.items()}
    
    # Find matching keys and check their indices
    for key in memory:
        if not key.startswith(f"{name}|"):
            continue
        # Get indices part if it exists
        key_parts = key.split('|')
        if len(key_parts) < 3:
            continue
        # Check if target indices are a subset of stored indices
        stored_indices = set(key_parts[2].split('_'))
        if target_indices.issubset(stored_indices) or target_indices.issubset(stored_indices):
            return memory[key]
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
