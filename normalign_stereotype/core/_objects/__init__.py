"""
Normalign NPC (Neural Processing Core) package.

This package provides core functionality for neural processing concepts, references,
inference, and planning in the Normalign framework.
"""

from normalign_stereotype.core._objects._concept import (
    Concept,
    CONCEPT_TYPE_CLASSIFICATION,
    CONCEPT_TYPE_JUDGEMENT,
    CONCEPT_TYPE_RELATION,
    CONCEPT_TYPE_OBJECT,
    CONCEPT_TYPE_SENTENCE,
    CONCEPT_TYPE_ASSIGNMENT,
    CONCEPT_TYPES
)

from normalign_stereotype.core._objects._reference import (
    Reference,
    cross_action,
    cross_product,
    element_action
)

from normalign_stereotype.core._objects._inference import Inference
from normalign_stereotype.core._objects._plan import Plan

from normalign_stereotype.core._objects._utils import (
    _get_initial_concepts,
    _build_concept_mappings,
    _build_dependency_graph,
    _topological_sort,
    _validate_topological_order,
    _process_input_data
)

__all__ = [
    # Concept related
    'Concept',
    'CONCEPT_TYPE_CLASSIFICATION',
    'CONCEPT_TYPE_JUDGEMENT',
    'CONCEPT_TYPE_RELATION',
    'CONCEPT_TYPE_OBJECT',
    'CONCEPT_TYPE_SENTENCE',
    'CONCEPT_TYPE_ASSIGNMENT',
    'CONCEPT_TYPES',
    
    # Reference related
    'Reference',
    'cross_action',
    'cross_product',
    'element_action',
    
    # Core classes
    'Inference',
    'Plan',
    
    # Utility functions
    '_get_initial_concepts',
    '_build_concept_mappings',
    '_build_dependency_graph',
    '_topological_sort',
    '_validate_topological_order',
    '_process_input_data'
] 