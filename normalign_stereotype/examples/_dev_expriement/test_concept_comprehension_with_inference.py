from normalign_stereotype.core._agent import Agent, get_default_working_config
from normalign_stereotype.core._concept import Concept, CONCEPT_TYPE_OBJECT, CONCEPT_TYPE_JUDGEMENT
from normalign_stereotype.core._inference import Inference
from normalign_stereotype.core._modified_llm import ConfiguredLLM, StructuredLLM, BulletLLM
from normalign_stereotype.core._reference import Reference


def test_judgement_concept_comprehension():
    # Initialize LLMs
    model_name = "qwen-plus"
    llm = ConfiguredLLM(model_name=model_name)
    structured_llm = StructuredLLM(model_name=model_name)
    bullet_llm = BulletLLM(model_name=model_name)
    
    # Create agent with LLMs
    agent = Agent({
        "llm": llm,
        "structured_llm": structured_llm,
        "bullet_llm": bullet_llm
    })
    
    # Define judgement concept
    judgement_name = "is_metaphor"
    judgement_context = "A metaphor is a figurative language element that directly maps a specific, tangible entity onto an abstract, complex theme if both components are present and the tangible entity transfers symbolic meaning to the abstract theme without intermediaries."
    
    # Create judgement concept
    judgement_concept = Concept(
        name=judgement_name,
        type=CONCEPT_TYPE_JUDGEMENT,
        reference={"context": judgement_context}
    )
    
    # Create inference for judgement
    inference = Inference(judgement_concept, agent)
    
    # Set up working configs
    perception_config = get_default_working_config(judgement_concept.comprehension["type"])
    actuation_config = get_default_working_config(judgement_concept.comprehension["type"])
    
    # Define inference
    inference.inference_definition(
        perception_concepts=[judgement_concept],
        actuation_concept=judgement_concept,
        perception_working_config=perception_config,
        actuation_working_config=actuation_config
    )
    
    # Execute inference
    result = inference.execute()
    
    # Print results
    print("Judgement Concept Comprehension Result:")
    print(result)

def test_object_concept_comprehension():
    # Initialize LLMs
    model_name = "qwen-plus"
    llm = ConfiguredLLM(model_name=model_name)
    structured_llm = StructuredLLM(model_name=model_name)
    bullet_llm = BulletLLM(model_name=model_name)
    
    # Create agent with LLMs
    agent = Agent({
        "llm": llm,
        "structured_llm": structured_llm,
        "bullet_llm": bullet_llm
    })
    
    # Define concepts
    concept_name = "specific_tangible_entity"
    concept_context = "From an extract, a metaphor is a figurative language element that directly maps a specific, tangible entity onto an abstract, complex theme if both components are present and the tangible entity transfers symbolic meaning to the abstract theme without intermediaries."
    
    # Create concept to infer
    concept_to_infer = Concept(
        name=concept_name,
        reference={"context": concept_context}
    )
    
    # Create inference
    inference = Inference(concept_to_infer, agent)
    
    # Set up working configs
    perception_config = get_default_working_config(concept_to_infer.comprehension["type"])
    actuation_config = get_default_working_config(concept_to_infer.comprehension["type"])
    
    # Define inference
    inference.inference_definition(
        perception_concepts=[concept_to_infer],
        actuation_concept=concept_to_infer,
        perception_working_config=perception_config,
        actuation_working_config=actuation_config
    )
    
    # Execute inference
    result = inference.execute()
    
    # Print results
    print("Object Concept Comprehension Result:")
    print(result)

if __name__ == "__main__":
    test_judgement_concept_comprehension()
    test_object_concept_comprehension() 