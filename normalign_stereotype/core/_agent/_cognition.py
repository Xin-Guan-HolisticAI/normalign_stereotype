import json
import logging
import re

def _remember_in_concept_name_location_dict(name, value, concept_name, memory_location, index_dict=None):
    """Persist data to JSON file using concept_name|name|location format"""
    # Create a key with pipe separator and sorted location info
    if index_dict:
        # Sort indices for consistent key format
        sorted_indices = '_'.join(f"{k}{v}" for k, v in sorted(index_dict.items()))
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
    target_indices = {f"{k}{v}" for k, v in indices.items()}
    
    # Find matching keys and check their indices
    for key in memory:
        if not key.startswith(f"{name}|"):
            continue
        # Get indices part if it exists
        key_parts = key.split('|')
        if len(key_parts) < 3:
            continue
        # Check if stored indices are a subset of target indices
        if set(key_parts[2].split('_')).issubset(target_indices):
            return memory[key]
    return None

def _cognition_memory_bullet(bullet, concept_name, memory_location, index_dict, remember):
    value, name = bullet.rsplit(':', 1)
    remember(name.strip(), value.strip(), concept_name, memory_location, index_dict)
    return name

def _cognition_memory_json_bullet(json_bullet, concept_name, memory_location, index_dict, remember):
    """Process JSON bullet points and update memory with location awareness"""
    try:
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
        
        # Update memory with index information
        remember(name, value.strip(), concept_name, memory_location, index_dict)
        return name
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {str(e)}")
        logging.error(f"JSON string: {json_bullet}")
        return None
    except Exception as e:
        logging.error(f"Error processing JSON bullet: {str(e)}")
        logging.error(f"JSON string: {json_bullet}")
        return None
