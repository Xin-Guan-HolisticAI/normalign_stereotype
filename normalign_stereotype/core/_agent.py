import json
import os
from normalign_stereotype.core._tools import LLMTool as LLM
import tempfile
from normalign_stereotype.core._reference import element_action
from normalign_stereotype.core._concept import Concept
import re


def get_default_working_config(concept_name):
    """Get default cognition configuration based on concept name."""
    perception_config = {
        "mode": "memory_retrieval"
    }

    place_holders = {
                "meta_input_name_holder": "{meta_input_name}",
                "meta_input_value_holder": "{meta_input_value}",
                "input_key_holder": "{input_name}",
                "input_value_holder": "{input_value}",
            }

    if "?" in concept_name:
        actuation_config = {
            "mode": "llm_prompt_two_replacement",
            "actuated_llm": "structured_llm",
            "prompt_template": classification_prompt,
            "place_holders": place_holders
        }
    elif "<" and ">" in concept_name:
        actuation_config = {
            "mode": "llm_prompt_two_replacement",
            "actuated_llm": "structured_llm",
            "prompt_template": judgement_prompt,
            "place_holders": place_holders
        }
    else:
        actuation_config = {
            "mode": "llm_prompt_two_replacement",
            "actuated_llm": "structured_llm",
            "prompt_template": concept_comprehension_prompt,
            "place_holders": place_holders
        }

    return perception_config, actuation_config


def customize_actuation_working_config(concept_name, prompt_template_dir=None, mode="llm_prompt_two_replacement", prompt_template=None, **kwargs):
    """Customize actuation working configuration.
    
    Args:
        concept_name: Name of the concept
        prompt_template_dir: Optional directory containing prompt templates
        mode: Mode of operation
        prompt_template: Optional direct prompt template string
        **kwargs: Additional configuration parameters
    """
    actuation_working_config = {
        "mode": mode,
        "place_holders": {
            "meta_input_name_holder": "{meta_input_name}",
            "meta_input_value_holder": "{meta_input_value}",
            "input_key_holder": "{input_name}",
            "input_value_holder": "{input_value}",
        }
    }

    if prompt_template:
        actuation_working_config["prompt_template"] = prompt_template
    elif prompt_template_dir:
        prompt_template_path = os.path.join(prompt_template_dir, concept_name)
        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
        actuation_working_config["prompt_template_path"] = prompt_template_path
    else:
        raise ValueError("Either prompt_template or prompt_template_dir must be provided")

    if mode == "llm_prompt_two_replacement":
        actuation_working_config["actuated_llm"] = "structured_llm"
    elif mode == "llm_prompt_generation_replacement":
        actuation_working_config["meta_llm"] = "llm"
        actuation_working_config["actuated_llm"] = "bullet_llm"

    actuation_working_config.update(kwargs)

    return actuation_working_config

class Agent:
    def __init__(self, body):
        self._validate_body(body)
        self.body = body
        self.working_memory = {
            'perception': {},
            'actuation': {},
        }

    def _validate_body(self, body):
        """Validate initialization parameters"""
        if 'llm' not in body or not isinstance(body['llm'], LLM):
            raise ValueError("Requires LLM instance in body")
        if 'memory_location' not in body or not os.path.exists(body['memory_location']):
            raise ValueError("Valid file path required for memory_location")

    def cognition(self, concept, mode = "memory_bullet", perception_working_config = None, actuation_working_config = None, **kwargs):
        """Process values into names and store"""

        if not isinstance(concept, Concept):
            raise ValueError("Perception requires Concept instance")

        raw_reference = concept.reference
        concept_name = concept.comprehension.get("name")

        if mode == "memory_bullet":
            self.working_memory['perception'][concept_name] = perception_working_config or {"mode": "memory_retrieval"}
            self.working_memory['actuation'][concept_name] = actuation_working_config or {"mode": "classification"}
            _cognition_memory_bullet_element = lambda bullet: self._cognition_memory_bullet(
                bullet,
                concept_name,
            )
            return element_action(_cognition_memory_bullet_element, [raw_reference])

        raise ValueError(f"Unknown cognition mode: {mode}")

    def _cognition_memory_bullet(self, bullet, concept_name):
        value, name = bullet.rsplit(':', 1)
        self._update_memory(name.strip(), value.strip(), concept_name)
        return name

    def _key_memory(self, name_may_list, concept_name_may_list):
        """Format name-concept pairs as key for searching in memory, handling both single values and lists"""

        # Both are lists - zip them together
        if isinstance(name_may_list, list) and isinstance(concept_name_may_list, list):
            if len(name_may_list) != len(concept_name_may_list):
                raise ValueError("name_may_list and concept_name_may_list must be same length when both are lists")
            return [f"{n} ({c})" for n, c in zip(name_may_list, concept_name_may_list)]

        # Name is list but concept is single - apply same concept to all names
        elif isinstance(name_may_list, list):
            return [f"{n} ({concept_name_may_list})" for n in name_may_list]

        # Concept is list but name is single - apply same name to all concepts
        elif isinstance(concept_name_may_list, list):
            return [f"{name_may_list} ({c})" for c in concept_name_may_list]

        # Both are single values - simple combination
        else:
            return f"{name_may_list} ({concept_name_may_list})"

    def _update_memory(self, name, value, concept_name):
        """Persist data to JSON file using JSON Lines format (one JSON object per line)"""

        _key_memory_concept = lambda x:self._key_memory(x, concept_name)

        with open(self.body['memory_location'], 'r+') as f:
            data = json.load(f)  # Reads and moves pointer to EOF
            data[_key_memory_concept(name)] = value  # Modify data
            f.seek(0)  # Move pointer back to start
            json.dump(data, f)  # Write new data (overwrites from position 0)
            f.truncate()

    def perception(self, concept):
        """Retrieve values through different perception modes"""

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

        perception_configuration = self.working_memory['perception']
        concept_configuration = perception_configuration.get(str(concept_name_may_list))
        _key_memory_concept = lambda x:self._key_memory(x, concept_name_may_list)

        mode = concept_configuration.get("mode")

        if mode == 'identity':
            _identity_perception = lambda name_may_list: (
                self._perception_identity(
                    _key_memory_concept(name_may_list)
                )
            )
            return element_action(_identity_perception, [reference])
        elif mode == 'memory_retrieval':
            _memory_retrieval_perception = lambda name_may_list:(
                self._perception_memory_retrieval(
                    _key_memory_concept(name_may_list)
                )
            )

            return element_action(_memory_retrieval_perception, [reference])
        elif mode == 'llm_generation':
            prompt_template = perception_configuration.get("prompt_template")
            llm_name = perception_configuration.get("llm")
            llm = self.body[llm_name]
            name_holder = perception_configuration.get("name_holder", "{input}")
            _llm_generation_perception = lambda name_may_list:(
                self._perception_llm_generation(
                    _key_memory_concept(name_may_list),
                    prompt_template,
                    llm,
                    name_holder,
                )
            )
            return element_action(_llm_generation_perception, [reference])
        raise ValueError(f"Unknown perception mode: {mode}")


    def _perception_identity(self, name_may_list):
        """Direct value return"""
        return [name_may_list, name_may_list]

    def _perception_memory_retrieval(self, name_may_list):
        """File-based value retrieval from JSONL storage"""
        with open(self.body['memory_location'], 'r', encoding='utf-8') as f:
            memory = eval(f.read())
            if isinstance(name_may_list,list):
                name_list = name_may_list
                value_list = []
                for name in name_list:
                    value = memory.get(name)
                    value_list.append(value)
                return [name_list, value_list]
            else:
                name = name_may_list
                value = memory.get(name)
                return [name, value]

    def _perception_llm_generation(self, name_may_list, prompt_template, llm, name_holder="{input}"):
        """LLM-processed value retrieval (supports single names or lists)"""
        if isinstance(name_may_list, list):
            value_list = []
            for name in name_may_list:
                prompt = prompt_template.replace(name_holder, name)
                value = llm.invoke(prompt)
                value_list.append(value)
            return [name_may_list, value_list]
        else:
            prompt = prompt_template.replace(name_holder, name_may_list)
            value = llm.invoke(prompt)
            return [name_may_list, value]


    def actuation(self, concept):
        """Create functions through named parameter resolution"""

        if not isinstance(concept, Concept):
            raise ValueError("Actuation requires Concept instance")

        reference = concept.reference
        concept_name = concept.comprehension.get("name","")
        concpet_context = concept.comprehension.get("context","")

        actuation_configuration = self.working_memory['actuation']
        concept_configuration = actuation_configuration.get(concept_name)

        _key_memory_concept = lambda x:self._key_memory(x, concept_name)
        mode = concept_configuration.get("mode")
     

        if mode == "classification":
            actuated_llm = self.body.get(concept_configuration.get('actuated_llm'))
            prompt_template = concept_configuration.get('prompt_template')
            if not prompt_template:
                prompt_template_path = concept_configuration.get('prompt_template_path')
                prompt_template = open(prompt_template_path, encoding="utf-8").read()
            place_holders = concept_configuration.get('place_holders')

            _classification_actuation = lambda name: (
                self._actuation_llm_prompt_two_replacement(
                name,
                prompt_template,
                place_holders,
                _key_memory_concept,
                actuated_llm,
            ))

            return element_action(_classification_actuation, [reference])

        if mode == "pos":
            actuated_llm = self.body.get(concept_configuration.get('actuated_llm'))
            meta_llm = self.body.get(concept_configuration.get('meta_llm'))
            prompt_template = concept_configuration.get('prompt_template')
            if not prompt_template:
                prompt_template_path = concept_configuration.get('prompt_template_path')
                prompt_template = open(prompt_template_path, encoding="utf-8").read()
            place_holders = concept_configuration.get('place_holders')

            _pos_actuation = lambda name: (
                self._actuation_llm_prompt_generation_replacement(
                name,
                prompt_template,
                place_holders,
                _key_memory_concept,
                meta_llm,
                actuated_llm,
            ))
            return element_action(_pos_actuation, [reference])
        
        if mode == "llm_prompt_generation_replacement":
            meta_prompt_llm = self.body.get(concept_configuration.get('meta_prompt_llm'))
            actuated_llm = self.body.get(concept_configuration.get('actuated_llm'))
            prompt_template = concept_configuration.get('prompt_template')
            if not prompt_template:
                prompt_template_path = concept_configuration.get('prompt_template_path')
                prompt_template = open(prompt_template_path, encoding="utf-8").read()
            place_holders = concept_configuration.get('place_holders')

            _llm_prompt_generation_replacement_actuation = lambda name: (
                self._actuation_llm_prompt_generation_replacement(
                    name,
                    prompt_template,
                    place_holders,
                    _key_memory_concept,
                    meta_prompt_llm,
                    actuated_llm,
                ))
            return element_action(_llm_prompt_generation_replacement_actuation, [reference])

        if mode == "llm_prompt_two_replacement":
            actuated_llm = self.body.get(concept_configuration.get('actuated_llm'))
            prompt_template = concept_configuration.get('prompt_template')
            if not prompt_template:
                prompt_template_path = concept_configuration.get('prompt_template_path')
                prompt_template = open(prompt_template_path, encoding="utf-8").read()
            place_holders = concept_configuration.get('place_holders')

            _classification_actuation = lambda name: (
                self._actuation_llm_prompt_two_replacement(
                name,
                prompt_template,
                place_holders,
                _key_memory_concept,
                actuated_llm,
            ))

            return element_action(_classification_actuation, [reference])


        raise ValueError(f"Unknown actuation mode: {mode}")


    def _clean_parentheses(self, text):
        # Remove parentheses content then clean up spaces
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


    def _actuation_llm_prompt_two_replacement(self, to_actuate_name, prompt_template, place_holders, key_build,
                                              actuated_llm, to_actuate_concept_name = ''):

        memory_location = self.body.get("memory_location")
        memory = eval(open(memory_location).read())

        meta_input_name_holder = place_holders.get("meta_input_name_holder", "{meta_input_name}")
        meta_input_value_holder = place_holders.get("meta_input_value_holder", "{meta_input_value}")
        input_key_holder = place_holders.get("input_key_holder", "{input_name}")
        input_value_holder = place_holders.get("input_value_holder", "{input_value}")

        to_actuate_value = memory.get(key_build(to_actuate_name), to_actuate_name)
        actuated_prompt = (prompt_template.replace(meta_input_name_holder, self._clean_parentheses(to_actuate_name)).
                           replace(meta_input_value_holder, to_actuate_value))

        def actuated_func(input_perception):
            input_key = input_perception[0]
            input_value = input_perception[1]
            passed_in_prompt = (actuated_prompt.replace(input_key_holder, self._clean_parentheses(str(input_key))).
                                replace(input_value_holder, str(input_value)))
            print("         passed in prompt:  ", repr(passed_in_prompt))
            return eval(actuated_llm.invoke(passed_in_prompt))

        return actuated_func


    def _actuation_llm_prompt_two_replacement(self, to_actuate_name, prompt_template, place_holders, key_build,
                                              actuated_llm):

        memory_location = self.body.get("memory_location")
        memory = eval(open(memory_location).read())

        meta_input_name_holder = place_holders.get("meta_input_name_holder", "{meta_input_name}")
        meta_input_value_holder = place_holders.get("meta_input_value_holder", "{meta_input_value}")
        input_key_holder = place_holders.get("input_key_holder", "{input_name}")
        input_value_holder = place_holders.get("input_value_holder", "{input_value}")

        to_actuate_value = memory.get(key_build(to_actuate_name), to_actuate_name)
        actuated_prompt = (prompt_template.replace(meta_input_name_holder, self._clean_parentheses(to_actuate_name)).
                           replace(meta_input_value_holder, to_actuate_value))

        def actuated_func(input_perception):
            input_key = input_perception[0]
            input_value = input_perception[1]
            passed_in_prompt = (actuated_prompt.replace(input_key_holder, self._clean_parentheses(str(input_key))).
                                replace(input_value_holder, str(input_value)))
            print("         passed in prompt:  ", repr(passed_in_prompt))
            return eval(actuated_llm.invoke(passed_in_prompt))

        return actuated_func

    # actuation function for name and actuation
    def _actuation_llm_prompt_generation_replacement(self, to_actuate_name, meta_prompt_template, place_holders,
                                                     key_build, meta_llm, actuated_llm):

        memory_location = self.body.get("memory_location")
        memory = eval(open(memory_location).read())

        meta_input_name_holder = place_holders.get("meta_input_name_holder", "{meta_input_name}")
        meta_input_value_holder = place_holders.get("meta_input_value_holder", "{meta_input_value}")
        input_key_holder = place_holders.get("input_key_holder", "{input_name}")
        input_value_holder = place_holders.get("input_value_holder", "{input_value}")

        to_actuate_value = memory.get(key_build(to_actuate_name), to_actuate_name)
        meta_prompt = (meta_prompt_template.replace(meta_input_name_holder, self._clean_parentheses(to_actuate_name))
                       .replace(meta_input_value_holder, to_actuate_value))
        actuated_prompt = meta_llm.invoke(meta_prompt)

        def actuated_func(input_perception):
            input_key = input_perception[0]
            input_value = input_perception[1]
            passed_in_prompt = (actuated_prompt.replace(input_key_holder, self._clean_parentheses(str(input_key)))
                                .replace(input_value_holder, str(input_value)))
            print("         passed in prompt:  ", repr(passed_in_prompt))
            return eval(actuated_llm.invoke(passed_in_prompt))

        return actuated_func


# Example usage
if __name__ == "__main__":
    pass

    # class MockLLM(LLM):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__('temp', {})
    #
    #     def invoke(self, prompt):
    #         """Always return valid Python code strings"""
    #         if "part of speech" in prompt:
    #             return "verb"  # Return string as valid Python string
    #         elif "meta-prompt-template" in prompt:
    #             input_value = prompt.split('|')[-1]
    #             input_name = input_value.split(':')[0]
    #             output_prompt = prompt.replace('meta-prompt-template', 'actuated') + '&{input}'
    #             output_prompt = output_prompt.replace(input_value, input_name)
    #             return output_prompt
    #         elif "actuated" in prompt:
    #             actuated = prompt.split('&')[0]
    #             input_value = prompt.split('&')[-1]
    #             output_value = [input_value.lower()+': '+ actuated, input_value.upper()+': '+ actuated]
    #             return str(output_value)
    #         return "lambda x: x"  # Return valid default function
    #
    # # Create temporary file for testing
    # with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
    #     tmp_path = tmpfile.name
    #
    # # Initialize cognitive agency
    # agency = Agent({
    #     'llm': MockLLM(),
    #     'file_location': tmp_path
    # })
    #
    # # Cognitive processing example
    # input_data = "greet:Hello world!"
    # name = agency.cognition(input_data)
    #
    #
    # print(name + ' ' + get_phrase_pos(name))

    # # Perception demonstration
    # print("Perception Results:")
    # print(f"Direct perception: {agency.perception(name)}")
    # print(f"File perception: {agency.perception(name)}")
    #
    # # Actuation demonstration
    # print("\nActuation Results:")
    # operation = agency.actuation(name)
    # print(f"Generated function output: {operation('test input')}")
    #
    # # Show stored data
    # print("\nStorage Contents:")
    # with open(tmp_path, 'r') as f:
    #     print(f.read())
    #
    # # Cleanup
    # os.unlink(tmp_path)