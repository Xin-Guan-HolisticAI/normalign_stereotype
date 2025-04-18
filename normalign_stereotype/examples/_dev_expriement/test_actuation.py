from normalign_stereotype.core._agent import AgentFrame
from normalign_stereotype.core._concept import Concept
from normalign_stereotype.core._reference import Reference, cross_action
from normalign_stereotype.core._modified_llm import ConfiguredLLM, JsonBulletLLM, JsonStructuredLLM
from typing import Optional, List, Union
import logging

class Inference:
    def __init__(
        self, 
        concept_to_infer: Concept,
        perception_concepts: Union[List[Concept], Concept],
        actuation_concept: Concept,
        view: Optional[List[str]] = None,
    ):
        """Initialize an Inference instance with all necessary components.
        
        Args:
            concept_to_infer: The concept to be inferred
            perception_concepts: List of Concept objects for perception
            actuation_concept: Single Concept object for actuation
            view: Optional list of axes to keep in the view
        """
        self.concept_to_infer: Concept = concept_to_infer
        self.agent: Optional[AgentFrame] = None
        self.view = view or []  # Direct list of axes to keep
        self.post_cognition_pre_perception_concepts = perception_concepts
        self.post_cognition_pre_actuation_concept = actuation_concept
        self.combined_pre_perception_concept: Optional[Concept] = None

        # Validate perception concept
        if not isinstance(perception_concepts, (list, tuple)):
            perception_concepts = [perception_concepts]

        if not all(isinstance(c, Concept) for c in perception_concepts):
            raise TypeError("All perception inputs must be Concept instances")

    def execute(self, agent: AgentFrame, cognition=True, perception_config_to_give=None, actuation_config_to_give=None, shape_view=True):
        """Execute pipeline with direct axis selection and optional custom configuration.
        
        Args:
            agent: AgentFrame instance required for execution
            cognition: Whether to apply cognition to the reference
            perception_config_to_give: Optional custom perception configuration
            actuation_config_to_give: Optional custom actuation configuration
            shape_view: Whether to apply view shaping to the reference
        """
        self.agent = agent  # Set the agent for this execution

        # Combine perception concepts using the utility function
        self.combined_pre_perception_concept = agent.perception_combination(self.post_cognition_pre_perception_concepts, agent)

        perception_ref = agent.perception(self.combined_pre_perception_concept)
        actuation_ref = agent.actuation(self.post_cognition_pre_actuation_concept)

        if agent.debug:
            logging.debug("===========================")
            logging.debug("Now processing inference execution: %s", self)
            logging.debug("     concept to infer %s", self.concept_to_infer.comprehension["name"])
            logging.debug("     perception %s", self.combined_pre_perception_concept.comprehension["name"])
            logging.debug("     actuation %s", self.post_cognition_pre_actuation_concept.comprehension["name"])

            logging.debug("!! cross-actioning references:")
            logging.debug("     actu: %s %s", actuation_ref.axes, actuation_ref.tensor)
            logging.debug("     perc %s %s", perception_ref.axes, perception_ref.tensor)

        self.concept_to_infer.reference = cross_action(
            actuation_ref,
            perception_ref,
            self.concept_to_infer.comprehension["name"]
        )

        if agent.debug:
            logging.debug(" raw_result %s %s", self.concept_to_infer.reference.axes, self.concept_to_infer.reference.tensor)

        if cognition:
            configs = {}
            if perception_config_to_give:
                configs["perception_working_config"] = perception_config_to_give
            if actuation_config_to_give:
                configs["actuation_working_config"] = actuation_config_to_give

            self.concept_to_infer.reference = self.agent.cognition(
                self.concept_to_infer,
                **configs
            )

        if shape_view:
            self.concept_to_infer.reference = self.concept_to_infer.reference.shape_view(self.view)

        return self.concept_to_infer

if __name__ == "__main__":
    # Create agent with memory location
    llm_model_name = "qwen-turbo-latest"

    agent = AgentFrame(
        body={
            'memory_location': 'memory.json',
            'llm': ConfiguredLLM(model_name=llm_model_name),
            "bullet_llm": JsonBulletLLM(model_name=llm_model_name),
            "structured_llm": JsonStructuredLLM(model_name=llm_model_name),
        },
        debug=True
    )

    # Define the perception concepts
    figurative_language_element = """'The pen trembled in the hand of the diplomat.'"""
    specific_entity = "pen trembled"
    abstract_theme = "Fragility of Peace"

    # Create raw perception concepts
    raw_perception_concepts: list[Concept] = []
    
    # Create figurative language element concept
    figurative_concept = Concept(
        name="figurative_language_element",
        context="",
        type="[]"
    )
    figurative_reference = Reference(
        axes=["figurative_language_element"],
        shape=(1,),
        initial_value="""[{"Explanation": "The pen trembled in the hand of the diplomat says something more about the diplomat nervousness and uncertainty, which makes the sentence more figurative.", "Summary_Key": "'that the pen trembled in the hand of the diplomat'"}]"""
    )
    figurative_concept.reference = figurative_reference
    raw_perception_concepts.append(figurative_concept)

    # Create specific entity concept
    entity_concept = Concept(
        name="specific_entity",
        context="",
        type="[]"
    )
    entity_reference = Reference(
        axes=["specific_entity"],
        shape=(1,),
        initial_value='[{"Explanation": "The trembling pen shows the nervousness and uncertainty of diplomat, and it is a specific entity.", "Summary_Key": "Trembling Pen"}]'
    )
    entity_concept.reference = entity_reference
    raw_perception_concepts.append(entity_concept)

    # Create abstract theme concept
    theme_concept = Concept(
        name="abstract_theme",
        context="",
        type="[]"
    )
    theme_reference = Reference(
        axes=["abstract_theme"],
        shape=(1,),
        initial_value='[{"Explanation": "As diplomat mentioned represents diplomatic peaceful relations, the trembling of pen leads to the nervousness and uncertainty of the diplomat then further leads to the abstract theme of fragility of peace.", "Summary_Key": "Fragility of Peace"}]'
    )
    theme_concept.reference = theme_reference
    raw_perception_concepts.append(theme_concept)

    # Process each raw perception concept through cognition
    pre_perception_concepts: list[Concept] = []
    for concept in raw_perception_concepts:
        processed_reference = agent.cognition(concept)
        concept.reference = processed_reference
        pre_perception_concepts.append(concept)

    # Create actuation concept
    raw_actuation_concept = Concept(
        name="{1}_maps_{2}_onto_{3}_directly",
        context="",
        type="<>"
    )
    raw_actuation_reference = Reference(
        axes=[raw_actuation_concept.comprehension["name"]],
        shape=(1,),
        initial_value='[{"Explanation": "The relation \\"{1}_maps_{2}_onto_{3}_directly\\" is true if and only if the following conditions are met: {1} must be a figurative language element that establishes a direct symbolic connection between {2}, a specific and tangible entity, and {3}, an abstract and complex theme, such that the meaning of {2} symbolically represents or conveys {3} without any intervening concepts or ambiguities obstructing the transfer of symbolic meaning. Additionally, the mapping must occur explicitly within a context where both {2} and {3} are clearly present and the symbolic relationship is unambiguous.", "Summary_Key": "{1}_maps_{2}_onto_{3}_directly"}]'
    )
    raw_actuation_concept.reference = raw_actuation_reference

    # Process actuation concept through cognition
    processed_actuation_reference = agent.cognition(raw_actuation_concept)
    raw_actuation_concept.reference = processed_actuation_reference

    # Create the concept to infer
    concept_to_infer = Concept(
        name="figurative_language_element_maps_specific_entity_onto_abstract_theme_directly",
        context="",
        type="^"
    )

    # Create and execute inference
    inference = Inference(
        concept_to_infer=concept_to_infer,
        perception_concepts=pre_perception_concepts,
        actuation_concept=raw_actuation_concept,
        view=["figurative_language_element"]
    )

    # Execute the inference pipeline
    concept_to_infer = inference.execute(agent)
    print("Inference result:", concept_to_infer.reference.tensor)
    print("Inference axes:", concept_to_infer.reference.axes)



