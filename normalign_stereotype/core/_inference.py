import os

from normalign_stereotype.core._reference import Reference, cross_action, cross_product, element_action
from normalign_stereotype.core._concept import Concept
from normalign_stereotype.core._pos_analysis import _get_phrase_pos
from normalign_stereotype.core._agent import Agent, get_default_working_config
from typing import Optional


class Inference:
    def __init__(self, concept_to_infer: Concept, agent: Agent):
        self.concept_to_infer: Concept = concept_to_infer
        self.agent: Agent = agent
        self.view = []  # Direct list of axes to keep
        self.perception_concepts = []
        self.the_perception_concept: Optional[Concept] = None
        self.the_actuation_concept: Optional[Concept] = None
        self.raw_ref: Optional[Reference] = None
        self.viewed_ref: Optional[Reference] = None
        self.configured_ref: Optional[Reference] = None
        self.perception_working_config_concept_to_infer: Optional[dict] = None
        self.actuation_working_config_concept_to_infer: Optional[dict] = None
        self.customized_actuation_config: bool = False

    def _combine_perception_concepts(self, perception_concepts):
        #use cross-product to make the only perception concept for processing
        the_perception_concept_name = (
            str([pc.comprehension["name"] for pc in perception_concepts])
            if len(perception_concepts) > 1
            else perception_concepts[0].comprehension["name"]
        )
        the_perception_reference = (
            cross_product(
                [pc.reference for pc in perception_concepts]
            )
            if len(perception_concepts) > 1
            else perception_concepts[0].reference
        )

        self.the_perception_concept = Concept(
            name = the_perception_concept_name,
            reference = the_perception_reference,
        )

        self.agent.working_memory['perception'][the_perception_concept_name] = \
            {"mode": "memory_retrieval"}


    def inference_definition(self, perception_concepts, actuation_concept, perception_working_config=None, actuation_working_config=None):
        """Store concepts for cross-action while deferring execution.

        Args:
            perception_concepts: List of Concept objects for perception
            actuation_concept: Single Concept object for actuation
        """
        # Validate perception concept
        if not isinstance(perception_concepts, (list, tuple)):
            perception_concepts = [perception_concepts]

        if not all(isinstance(c, Concept) for c in perception_concepts):
            raise TypeError("All perception inputs must be Concept instances")

        # Validate actuation concept
        if not isinstance(actuation_concept, Concept):
            raise TypeError("Actuation input must be a single Concept instance")

        # Store concepts for later execution

        self.perception_concepts = perception_concepts
        self.the_actuation_concept = actuation_concept
        # self._combine_perception_concepts(self.perception_concepts)

        if perception_working_config:
            self.perception_working_config_concept_to_infer = perception_working_config
        if actuation_working_config:
            self.actuation_working_config_concept_to_infer = actuation_working_config

    def view_definition(self, axes_list):
        """Directly set which axes to keep in the view"""
        if not isinstance(axes_list, list):
            raise TypeError("View must be a list of axes")
        self.view = axes_list

    def view_change(self):
        """Apply view by selecting specified axes, using all when empty"""
        if not self.concept_to_infer.reference:
            raise ValueError("Concept reference not defined - make a reference first")

        # Use all axes if view is empty
        selected_axes = self.view if self.view else self.concept_to_infer.reference.axes.copy()

        # Validate existence of selected axes
        available_axes = set(self.concept_to_infer.reference.axes)
        for axis in selected_axes:
            if axis not in available_axes:
                raise ValueError(f"Axis '{axis}' not found in reference axes")

        # Create new reference with selected axes
        self.viewed_ref = self.concept_to_infer.reference.slice(*selected_axes)
        return self.viewed_ref

    def execute(self, perception_config=None, actuation_config=None):
        """Execute pipeline with direct axis selection and optional custom configuration.
        
        Args:
            perception_config: Optional custom perception configuration
            actuation_config: Optional custom actuation configuration
        """
        if not hasattr(self, 'the_perception_concept') or not hasattr(self, 'the_actuation_concept'):
            raise ValueError("Define concepts first with inference_definition()")

        agent = self.agent

        self._combine_perception_concepts(self.perception_concepts)
        perception_ref = agent.perception(self.the_perception_concept)
        actuation_ref = agent.actuation(self.the_actuation_concept)

        print("===========================")
        print("Now processing inference execution:", self)
        print("     concept to infer", self.concept_to_infer.comprehension["name"])
        print("     perception", self.the_perception_concept.comprehension["name"])
        print("     actuation", self.the_actuation_concept.comprehension["name"])

        print("!! cross-actioning references:")
        print("     actu:", actuation_ref.axes, actuation_ref.tensor)
        print("     perc", perception_ref.axes, perception_ref.tensor)
        self.raw_ref = cross_action(
            actuation_ref,
            perception_ref,
            self.concept_to_infer.comprehension["name"]
        )
        print(" raw_result", self.raw_ref.axes, self.raw_ref.tensor)
        self.concept_to_infer.reference = self.raw_ref

        # Use custom config if provided by the class or the method, otherwise get default
        if perception_config is None:
            if self.perception_working_config_concept_to_infer is None:
                default_perception, default_actuation = get_default_working_config(
                    self.concept_to_infer.comprehension["name"]
                )
                self.perception_working_config_concept_to_infer = perception_config or default_perception
        else:
            self.perception_working_config_concept_to_infer = perception_config
        
        if actuation_config is None:
            if self.actuation_working_config_concept_to_infer is None:
                default_perception, default_actuation = get_default_working_config(
                    self.concept_to_infer.comprehension["name"]
                )
                self.actuation_working_config_concept_to_infer = actuation_config or default_actuation
        else:
            self.actuation_working_config_concept_to_infer = actuation_config


        self.concept_to_infer.reference = self.agent.cognition(
            self.concept_to_infer,
            perception_working_config=self.perception_working_config_concept_to_infer,
            actuation_working_config=self.actuation_working_config_concept_to_infer
        )

        self.view_change()
        self.concept_to_infer.reference = self.viewed_ref

        return self.concept_to_infer

    def cognition_configuration(self, execution = True):
        """Configure perception and actuation for the concept"""

        concept_name = self.concept_to_infer.comprehension["name"]

        self.perception_working_config_concept_to_infer = {
            "mode": "memory_retrieval"
        }

        if not self.customized_actuation_config:
            if "classification" in concept_name:
                self.actuation_working_config_concept_to_infer = {
                    "mode": "classification",
                    "actuated_llm": "structured_llm",
                    "prompt_template_path": os.path.join(PROJECT_ROOT, "normalign_stereotype/templates/basic_template/classification-d"),
                    "place_holders": {
                        "meta_input_name_holder": "{meta_input_name}",
                        "meta_input_value_holder": "{meta_input_value}",
                        "input_key_holder": "{input_name}",
                        "input_value_holder": "{input_value}",
                    },
                }
            else:
                pos = _get_phrase_pos(concept_name)
                input_key_holder = "{input_name}"
                if pos == "noun":
                    input_key_holder = "{verb/proposition}"
                elif pos == "verb":
                    input_key_holder = "{noun/object/event}"

                self.actuation_working_config_concept_to_infer = {
                    "mode": "pos",
                    "actuated_llm": "bullet_llm",
                    "meta_llm": "llm",
                    "prompt_template_path": os.path.join(PROJECT_ROOT, f"normalign_stereotype/templates/pos_template/{pos}"),
                    "place_holders": {
                        "meta_input_name_holder": "{meta_input_name}",
                        "meta_input_value_holder": "{meta_input_value}",
                        "input_key_holder": input_key_holder,
                        "input_value_holder": "{input_value}",
                    },
                }
        if execution:
            self.configured_ref = self.agent.cognition(
                self.concept_to_infer,
                perception_working_config=self.perception_working_config_concept_to_infer,
                actuation_working_config=self.actuation_working_config_concept_to_infer
            )
            return self.configured_ref

    def get_active_reference(self):
        """Get the currently active reference"""
        return self.configured_ref or self.viewed_ref or self.raw_ref