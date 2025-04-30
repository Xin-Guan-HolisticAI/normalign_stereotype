from ._utils import (
    _safe_eval,
    _format_bullet_points,
    _replace_placeholders_with_values,
    _clean_parentheses,
    _prompt_template_dynamic_substitution,
    _get_default_working_config
)

from ._cognition import (
    _remember_in_concept_name_location_dict,
    _recollect_by_concept_name_location_dict,
    _cognition_memory_bullet,
    _cognition_memory_json_bullet,
    _combine_pre_perception_concepts_by_two_lists
)

from ._actuation import (
    _actuation_llm_prompt_two_replacement
)

from ._perception import (
    _perception_memory_retrieval,

)

from normalign_stereotype.core._objects._reference import Reference, element_action
from normalign_stereotype.core._objects._concept import Concept
import logging

class AgentFrame:
    def __init__(self, body, mode_of_remember="memory_json_bullet", mode_of_recollection="concept_name_location_dict",  mode_of_perception_combination="two_lists", debug=False):
        self.body = body
        self.debug = debug
        self.working_memory = {
            'perception': {},
            'actuation': {},
            'cognition': {
                'mode_of_remember': mode_of_remember,
                'mode_of_recollection': mode_of_recollection,
                'mode_of_perception_combination': mode_of_perception_combination
            }
        }
        if mode_of_perception_combination == "two_lists":
            self.perception_combination = _combine_pre_perception_concepts_by_two_lists
        else:
            raise ValueError(f"Unknown perception combination mode: {mode_of_perception_combination}")
        
        # Set up logging if debug is enabled
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logging.debug(f"Initializing AgentFrame with mode_of_remember={mode_of_remember}, mode_of_recollection={mode_of_recollection}")
        
        # Set the recollection function based on mode
        if mode_of_recollection == "concept_name_location_dict":
            self.recollection = _recollect_by_concept_name_location_dict
            self.remember_in = _remember_in_concept_name_location_dict
            if debug:
                logging.debug("Using concept_name_location_dict for recollection")
        else:
            raise ValueError(f"Unknown recollection mode: {mode_of_recollection}")

    def cognition(self, concept, perception_working_config=None, actuation_working_config=None, **kwargs):
        """Process values into names and store"""
        if self.debug:
            logging.debug(f"Starting cognition process for concept: {concept.comprehension.get('name')}")
            
        if not isinstance(concept, Concept):
            raise ValueError("Perception requires Concept instance")

        raw_reference = concept.reference
        concept_name = concept.comprehension.get("name")
        concept_context = concept.comprehension.get("context")
        concept_type = concept.comprehension.get("type")

        if self.debug:
            logging.debug(f"Processing concept: name={concept_name}, type={concept_type}")

        default_perception_working_config, default_actuation_working_config = _get_default_working_config(concept_type)
        self.working_memory['perception'][concept_name] = perception_working_config or default_perception_working_config
        self.working_memory['actuation'][concept_name] = actuation_working_config or default_actuation_working_config

        if self.debug:
            logging.debug(f"Working memory updated for {concept_name}")

        if self.working_memory['cognition']['mode_of_remember'] == "memory_bullet":
            if self.debug:
                logging.debug("Using memory_bullet mode for cognition")
            _cognition_memory_bullet_element = lambda bullet, index_dict: _cognition_memory_bullet(
                bullet,
                concept_name,
                self.body['memory_location'],
                index_dict,
                self.remember_in
            )
            return element_action(_cognition_memory_bullet_element, [raw_reference], index_awareness=True)
        elif self.working_memory['cognition']['mode_of_remember'] == "memory_json_bullet":
            if self.debug:
                logging.debug("Using memory_json_bullet mode for cognition")
            _cognition_memory_json_bullet_element = lambda json_bullet, index_dict: _cognition_memory_json_bullet(
                json_bullet,
                concept_name,
                self.body['memory_location'],
                index_dict,
                self.remember_in
            )
            return element_action(_cognition_memory_json_bullet_element, [raw_reference], index_awareness=True)

        raise ValueError(f"Unknown cognition mode: {self.working_memory['cognition']['mode_of_remember']}")

    def perception(self, concept):
        """Retrieve values through different perception modes"""
        if self.debug:
            logging.debug(f"Starting perception process for concept: {concept.comprehension.get('name')}")
            
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

        if self.debug:
            logging.debug(f"Processing concept name: {concept_name_may_list}")

        perception_configuration = self.working_memory['perception']
        concept_configuration = perception_configuration.get(str(concept_name_may_list))

        mode = concept_configuration.get("mode")

        if self.debug:
            logging.debug(f"Using perception mode: {mode}")

        if mode == 'memory_retrieval':
            _memory_retrieval_perception = lambda name_may_list, index_dict:(
                _perception_memory_retrieval(
                    name_may_list,
                    concept_name_may_list,
                    index_dict,
                    self.recollection,
                    self.body['memory_location'],
                    self.debug,
                )
            )
            return element_action(_memory_retrieval_perception, [reference], index_awareness=True)
        
        raise ValueError(f"Unknown perception mode: {mode}")

    def actuation(self, concept, for_perception_concept_name=''):
        """Create functions through named parameter resolution"""
        if self.debug:
            logging.debug(f"Starting actuation process for concept: {concept.comprehension.get('name')}")
            
        if not isinstance(concept, Concept):
            raise ValueError("Actuation requires Concept instance")

        reference = concept.reference
        concept_name = concept.comprehension.get("name","")
        concept_context = concept.comprehension.get("context","")
        concept_type = concept.comprehension.get("type","{}")

        if self.debug:
            logging.debug(f"Processing actuation for concept: name={concept_name}, type={concept_type}")

        actuation_working_configuration = self.working_memory['actuation'].get(concept_name)
        mode = actuation_working_configuration.get("mode")

        if self.debug:
            logging.debug(f"Using actuation mode: {mode}")

        if mode == "llm_prompt_two_replacement":
            actuated_llm = self.body[actuation_working_configuration.get("llm")]
            prompt_template = actuation_working_configuration.get("prompt_template")
            variable_definitions = actuation_working_configuration.get("template_variable_definition_dict")

            if self.debug:
                logging.debug(f"LLM actuation parameters: prompt_template={prompt_template}, variable_definitions={variable_definitions}")

            _actuated_func = lambda actu_name, index_dict: (
                _actuation_llm_prompt_two_replacement(
                to_actuate_name=actu_name,
                prompt_template=prompt_template,
                variable_definitions=variable_definitions,
                actuated_llm=actuated_llm,
                to_actuate_concept_name=concept_name,
                perception_concept_name=for_perception_concept_name,
                index_dict=index_dict,
                recollection=self.recollection
                )
            )

            return element_action(_actuated_func, [reference], index_awareness=True)

        raise ValueError(f"Unknown actuation mode: {mode}")