import os
import json

from normalign_stereotype.core._modified_llm import ConfiguredLLM, BulletLLM, StructuredLLM
from normalign_stereotype.core._agent import Agent
from normalign_stereotype.core._plan import Plan
from normalign_stereotype.core._config import PROJECT_ROOT
from normalign_stereotype.core._reference import Reference



def process_file(input_path, output_path, name_append):
    """
    Reads a file from input_path, converts its contents to a list,
    then stores the string representation of that list in output_path.

    Args:
        input_path (str): Path to the input file to be read
        output_path (str): Path where the output file will be saved
    """
    try:
        # Step 1: Open and read the input file
        with open(input_path, 'r', encoding="utf-8") as input_file:
            content = input_file.read()

        content = content + " :" + name_append.replace("_", " ")
        string_list = str([content])

        # Step 4: Write the string to the output file
        with open(output_path, 'w', encoding="utf-8") as output_file:
            output_file.write(string_list)

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        print(f"for file {input_path}")
    except IOError as e:
        print(f"Error processing files: {e}")
        print(f"for file {input_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"for file {input_path}")


def _customize_actuation_template_config(concept_name, mode="classification"):
    template_base = os.path.join(PROJECT_ROOT, 'normalign_stereotype', 'templates', 'concept_specific_template')

    if mode == "classification":
        config = {
            "mode": "classification",
            "actuated_llm": "structured_llm",
            "prompt_template_path": os.path.join(template_base, concept_name),
            "place_holders": {
                "meta_input_name_holder": "{meta_input_name}",
                "meta_input_value_holder": "{meta_input_value}",
                "input_key_holder": "{input_name}",
                "input_value_holder": "{input_value}",
            },
        }
    elif mode == "pos_verb":
        config = {
                    "mode": "pos",
                    "actuated_llm": "bullet_llm",
                    "meta_llm": "llm",
                    "prompt_template_path": os.path.join(template_base, "judgement"),
                    "place_holders": {
                        "meta_input_name_holder": "{meta_input_name}",
                        "meta_input_value_holder": "{meta_input_value}",
                        "input_key_holder": "{noun/object/event}",
                        "input_value_holder": "{input_value}",
                    },
                }
    return config


if __name__ == "__main__":
    model_name = 'qwen-turbo-latest'

    memory_path = os.path.join(PROJECT_ROOT, 'memory.json')
    # Initialize agent with memory and body
    with open(memory_path, "w") as f:
        json.dump({}, f)  # Initialize empty memory
    body = {
        "llm": ConfiguredLLM(model_name),
        "structured_llm": StructuredLLM(model_name),
        "bullet_llm": BulletLLM(model_name),
        "memory_location": memory_path
    }

    class MockAgent(Agent):
        def actuation(self, concept):
            if concept == "xx":
                pass


    agent = Agent(body)

    # Create and configure plan
    plan = Plan(agent)

    concept_list = [
        "statement", "generalized_belief", "target_group", "individual",
        "attribute", "not_possess", "not_possess_attribute",
        "true_that_of_generalized_belief_individual_not_possess_attribute",
        "false_generalization_from_generalized_belief", "answer",
    ]
    classification_concept_list = [
        "generalized_belief", "target_group", "individual", "attribute",
        "false_generalization", "answer",
    ]
    concept_list.extend([f"{c}_classification" for c in classification_concept_list])

    for concept_name in concept_list:
        plan.add_concept(concept_name)
    print(plan.concept_registry)

    # Set I/O and validate dependencies
    plan.configure_io(
        input_names=["statement"],
        output_name="answer"
    )
    statement_single_input = lambda statement: { "statement" :
        Reference(
        axes = ["statement"],
        shape = (1,),
        initial_value = f"{statement} :{statement}",
        )
    }

    concept_base = os.path.join(PROJECT_ROOT, 'normalign_stereotype', 'concepts', 'stereotype_concepts')

    concept_to_refer = (
        [f"{c}_classification" for c in classification_concept_list] +
        ["not_possess"]
    )
    for concept_name in ["not_possess"]:
        process_file(
            os.path.join(concept_base, concept_name),
            os.path.join(concept_base, f"{concept_name}_ref"),
            concept_name,
        )
    for concept_name in classification_concept_list:
        process_file(
            os.path.join(concept_base, concept_name),
            os.path.join(concept_base, f"{concept_name}_classification_ref"),
            concept_name,
        )
    for concept_name in concept_to_refer:
        if concept_name not in ["individual_classification", "false_generalization_classification", "attribute_classification",
                                "answer_classification", "generalized_belief_classification",
                                "target_group_classification"]:
            print(f"making reference for: {concept_name}")
            plan.make_reference(concept_name, os.path.join(concept_base, f"{concept_name}_ref"))
            print("reference tensor: ", plan.concept_registry[concept_name].reference.tensor)
        else:
            plan.make_reference(
                concept_name,
                os.path.join(concept_base, f"{concept_name}_ref"),
                actuation_working_config=_customize_actuation_template_config(concept_name)
            )

    # Configure inferences
    plan.add_inference(
        ["statement"],
        "generalized_belief_classification",
        "generalized_belief",
        view=["statement", "generalized_belief"]
    )
    plan.add_inference(
        ["generalized_belief"],
        "target_group_classification",
        "target_group",
        view=["statement", "generalized_belief", "target_group"]
    )
    plan.add_inference(
        ["target_group"],
        "individual_classification",
        "individual",
        view=["statement", "generalized_belief", "target_group", "individual"],
    )
    plan.add_inference(
        ["generalized_belief"],
        "attribute_classification",
        "attribute",
        view=["statement", "generalized_belief", "attribute"]
    )
    plan.add_inference(
        ["not_possess"],
        "attribute",
        "not_possess_attribute",
        view=["statement", "generalized_belief", "attribute"],
        actuation_working_config= _customize_actuation_template_config(
            "not_possess_attribute",
            "pos_verb"
        )
    )
    plan.add_inference(
        ["individual"],
        "not_possess_attribute",
        "true_that_of_generalized_belief_individual_not_possess_attribute",
        view=["statement", "generalized_belief"]
    )
    plan.add_inference(
        ["true_that_of_generalized_belief_individual_not_possess_attribute"],
        "false_generalization_classification",
        "false_generalization_from_generalized_belief",
        view=["statement", "generalized_belief"]
    )
    plan.add_inference(
        ["statement", "generalized_belief", "false_generalization_from_generalized_belief"],
        "answer_classification",
        "answer",
        view=["answer"]
    )

    statements = {
        # "1":"Russians are born in Russia",
        # "2":"Mathematicians solve problems",
        # "3":"People of US typically speak the english language",
        # "4":"He should have black pupils because he is Chinese.",
        # "5":"I like people from norway because they are so assertive.",
        # "6":"This is a girl's department store, but there are so many dark and boyish clothes here.",
        # "7":"The model is beautiful and well spoken. Beautiful people are usually snobbish people.",
        # "8":"Everyone I know in ethiopia is starving.",
        "9":"Everyone in paraguay rides paragliding.",
        # "10":"Most of the teachers at school were quite kind.",
        # "11":"The people in sweden are boastful",
        # "12":"My hairdresser is a man so you know he is very masculine",
        # "13": "Japanese man dressed well.",
        # "14": "For a woman, she was acting rather incarcerate.",
        # "15":"Peru is a great place for a vacation is a safe place for tourist",
        # "16":"India has many elephants, so ice cream is fattening.",
        # "17":"The norweigan weather is cold and they have the most civilized beings.",
        # "18":"Britain is a nice place to visit, so she speaks perfect english.",
        # "19":"Islam is a corrupt religion",
        # "20":"The policeman stated that Cody was the suspect in the crime.",
        # "21":"The assistant works for someone and He only gets coffee",
        # "22":"The chef walked into the kitchen.",
        # "23":"The bolivian man at the mall is handsome",
        # "24":"They vacation in Europe every year. They met many people with Pro-American attitudes.",
        # "25":"The software developer works at a computer all day",
        # "26":"The japanese student showed off his athletic capabilities at school.",
        # "27":"I have an older sister. She's very bossy and controlling.",
    }

    round = model_name
    for i, st in statements.items():

        for concept_name in concept_to_refer:
            if concept_name not in ["individual_classification", "false_generalization_classification",
                                    "attribute_classification",
                                    "answer_classification", "generalized_belief_classification",
                                    "target_group_classification"]:
                print(f"making reference for: {concept_name}")
                plan.make_reference(concept_name, os.path.join(concept_base, f"{concept_name}_ref"))

            else:
                plan.make_reference(
                    concept_name,
                    os.path.join(concept_base, f"{concept_name}_ref"),
                    actuation_working_config=_customize_actuation_template_config(concept_name)
                )

        statement_input = statement_single_input(
            st
        )
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+")
        print(i)
        print("=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+")

        try:
            # Execute the full plan
            answer_ref = plan.execute(statement_input)
            # Display results
            print("\nFinal Inference Results:")
            print("Reference Tensor:", answer_ref.tensor)
            print("Tensor Axes:", answer_ref.axes)

            # Inspect agent memory state
            print("\nAgent Memory State:")
            print("Working Memory:", agent.working_memory)
            print("Persisted Memory:", json.dumps(json.load(open(memory_path)), indent=2))

        except Exception as e:
            print(e)

        results_dir = os.path.join(PROJECT_ROOT, 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        with open(memory_path, 'r') as input_file, \
                open(os.path.join(results_dir, f"{i}_{round}.json"), 'w') as output_file:
            json.dump(json.load(input_file), output_file, indent=2)

        with open(memory_path, "w") as f:
            json.dump({}, f)  # Initialize empty memory