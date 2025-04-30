from normalign_stereotype.core._llm_tools import ConfiguredLLM, BulletLLM, StructuredLLM
from normalign_stereotype.core._objects._reference import Reference, cross_action, cross_product, element_action
from normalign_stereotype.core._objects._concept import Concept
from normalign_stereotype.core._agent_frame import Agent


if __name__ == '__main__':
    structured_llm = StructuredLLM()
    llm = ConfiguredLLM()
    bullet_llm = BulletLLM()

    working_memory = {}

    #initiate the target group reference
    target_group_ref_tensor = eval(open("stereotype_concepts/target_group_ref", encoding="utf-8").read())
    target_group_ref = Reference(
        axes=["target_group"],
        shape=(len(target_group_ref_tensor),),
        initial_value=0,
    )
    target_group_ref.tensor = target_group_ref_tensor
    print(target_group_ref.tensor)

    #initiate the individual classification reference
    individual_classification_ref_tensor = ["possible individuals"]
    individual_classification_ref = Reference(
        axes=["possible individuals classification"],
        shape=(len(individual_classification_ref_tensor),),
        initial_value=0,
    )
    individual_classification_ref.tensor = individual_classification_ref_tensor
    print(individual_classification_ref.tensor)

    #cognition for "possible individuals"
    cognition_name = "possible individuals"
    cognition_value = open("stereotype_concepts/individuals", encoding="utf-8").read()
    working_memory[cognition_name] = cognition_value
    cognition_name = "engineers(target group)"
    cognition_value = "Engineers are problem-solving professionals who design, build, and optimize systems, structures, and technologies to improve the world around us."
    working_memory[cognition_name] = cognition_value

    #actuation function for classification and actuation
    def classification_actuation(to_actuate_name, working_memory=working_memory, structured_llm=structured_llm):
        classify_d_template = open("../basic_template/classification-d", encoding="utf-8").read()
        to_actuate_value = working_memory.get(to_actuate_name, to_actuate_name)
        actuated_prompt = classify_d_template.replace("{meta_input_name}", to_actuate_name).replace(
            "{meta_input_value}", to_actuate_value)
        def actuated_func(input_name, working_memory=working_memory, actuated_prompt=actuated_prompt, structured_llm=structured_llm):
            input_value = working_memory.get(input_name, input_name)
            passed_in_prompt = actuated_prompt.replace("{input_name}", input_name).replace("{input_value}", input_value)
            return eval(structured_llm.invoke(passed_in_prompt))
        return actuated_func
    individual_classification_ref_actuated = element_action(classification_actuation, [individual_classification_ref])
    print(individual_classification_ref_actuated.tensor)

    #execution of assignment
    individual_ref_raw = cross_action(individual_classification_ref_actuated, target_group_ref, "possible individuals")
    print(individual_ref_raw.tensor)
    print(individual_ref_raw.shape)
    print(individual_ref_raw.axes)

    #apply cognition to individual_ref
    def cognition(name_value, concept_name, working_memory=working_memory):
        name, value = name_value.split(":", 1)
        name_concept = name + "(" + concept_name + ")"
        working_memory[name_concept] = value
        return name
    cognition_individual = lambda x: cognition(x, concept_name="possible individuals")
    individual_ref = element_action(cognition_individual, [individual_ref_raw])
    print(individual_ref.tensor)
    print(working_memory)

    # actuation function for name and actuation
    def pos_actuation(to_actuate_name, pos, concept_name, working_memory=working_memory, bullet_llm=bullet_llm, llm=llm):
        prompt_meta = open(f"normalign_stereotype/templates/pos_template/{pos}", encoding = "utf-8").read()
        key = to_actuate_name + "(" + concept_name + ")"
        to_actuate_value = working_memory.get(key, to_actuate_name)
        prompt_meta = prompt_meta.replace("{meta_input_name}", to_actuate_name).replace("{meta_input_value}", to_actuate_value)
        prompt_actuate = llm.invoke(prompt_meta)
        if pos == "noun":
            name_place_holder = "{verb/proposition}"
        elif pos == "verb":
            name_place_holder = "{noun/object/event}"
        def actuated_func(input_name_concept, name_place_holder = name_place_holder, working_memory=working_memory, actuated_prompt=prompt_actuate,
                          bullet_llm=bullet_llm):
            input_value = working_memory.get(input_name_concept, input_name_concept)
            passed_in_prompt = actuated_prompt.replace(name_place_holder, input_name_concept).replace("{input_value}", input_value)
            return eval(bullet_llm.invoke(passed_in_prompt))
        return actuated_func
    individual_pos_actuation = lambda x: pos_actuation(x, pos="noun", concept_name = "possible individuals")
    individual_ref_actuated = element_action(individual_pos_actuation, [individual_ref])

    #initiate to_tensor
    to_ref_tensor = ["to"]
    to_ref = Reference(
        axes=["to"],
        shape=(len(to_ref_tensor),),
        initial_value=0,
    )
    to_ref.tensor = to_ref_tensor
    print(to_ref.tensor)

    #acting_on_to_tensor
    to_individual_ref_raw = cross_action(individual_ref_actuated, to_ref, "to individuals")
    print(to_individual_ref_raw.tensor)
    print(to_individual_ref_raw.shape)

    #cognition to individual
    cognition_to_individual = lambda x: cognition(x, concept_name="to individual")
    to_individual_ref = element_action(cognition_to_individual, [to_individual_ref_raw])
    print(to_individual_ref.tensor)
    print(working_memory)


    #initiate apply_tensor
    apply_ref_tensor_raw = eval(open("stereotype_concepts/applying", encoding="utf-8").read())
    apply_ref_raw = Reference(
        axes=["apply"],
        shape=(len(apply_ref_tensor_raw),),
        initial_value=0,
    )
    apply_ref_raw.tensor = apply_ref_tensor_raw
    print(apply_ref_raw.tensor)

    #cognition apply
    cognition_apply = lambda x: cognition(x, concept_name="apply")
    apply_ref = element_action(cognition_apply, [apply_ref_raw])
    print(apply_ref.tensor)
    print(working_memory)

    #actuate_to_individual_tensor
    to_individual_pos_actuation = lambda x: pos_actuation(x, pos="verb", concept_name = "to individuals")
    to_individual_ref_actuated = element_action(to_individual_pos_actuation, [to_individual_ref])

    #acting on apply
    to_individual_apply_ref_raw = cross_action(to_individual_ref_actuated, apply_ref, "to individual apply")
    print(to_individual_apply_ref_raw.tensor)
    print(to_individual_apply_ref_raw.shape)











