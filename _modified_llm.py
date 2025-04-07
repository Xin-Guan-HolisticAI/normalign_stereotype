import re
import ast
from typing import List
import logging
from _tools import LLMTool as LLM

class ConfiguredLLM(LLM):
    def __init__(self, model_name = "deepseek-r1-distill-qwen-1.5b",max_retries=5, *args, **kwargs):
        super().__init__('temp', {}, model_name)
        self.max_retries = max_retries

    def invoke(self, user_input, max_retries = None):
        if max_retries:
            pass
        else:
            max_retries = self.max_retries

        return self._invoke(
                    prompt=user_input,
                    temperature=0,
                )


class BulletLLM(LLM):
    def __init__(self, model_name = "deepseek-r1-distill-qwen-1.5b", max_retries=5, *args, **kwargs):
        super().__init__('temp', {}, model_name)
        self.max_retries = max_retries

    def bullet_invoke(self, user_input: str, max_retries: int = 3) -> str:
        """
        Enhanced invoke with robust format validation and retries.
        Returns empty list if format validation fails after retries.
        """
        system_prompt = """To Answer the question, Format output ONLY as ONE point where a key is appended after answer:
           1. Start with full long explanation and reasoning from contextual information before colon. This should be clear and faithful to the context.
           2. Key = Short phrase Key summarizing the aspect. This Key should not contain information in bracket i.e. no '(word)'. 
           3. Format: 'Explanation... :Key' Example: 'Engineers are all doing engineering jobs ...: Engineers do Engineer Jobs'
           4. Make sure only one ":" is used overall.
           """

        for _ in range(max_retries):
            try:
                # Call base LLM implementation
                raw_output = super()._invoke(
                    prompt=user_input,
                    system_prompt=system_prompt,
                    temperature=0,
                )

                if not (isinstance(raw_output, str) and ':' in raw_output):
                    raise ValueError(f"Invalid raw output: {raw_output}")

                return raw_output.replace("- ","").replace("/n","")

            except (SyntaxError, ValueError, AttributeError, TypeError) as e:
                logging.warning(f"Validation failed: {str(e)}")
                continue

        return 'NULL'

    def invoke(self, user_input: str, max_retries = None):

        if max_retries:
            pass
        else:
            max_retries= self.max_retries

        bullet = self.bullet_invoke(user_input, max_retries)
        return str([bullet])



class StructuredLLM(LLM):
    def __init__(self,  model_name = "deepseek-r1-distill-qwen-1.5b",max_retries=5, *args, **kwargs):
        super().__init__('temp', {}, model_name)
        self.max_retries = max_retries

    def structured_invoke(self, user_input: str, max_retries: int = 3) -> List[str]:
        """
        Enhanced invoke with robust format validation and retries.
        Returns empty list if format validation fails after retries.
        """
        system_prompt = """Answer the question. If there are distinct elements of your answer, format output ONLY as a Python list: ["Key: Full explanation...", ...] where:

1. Explanation Requirements:
   - Start with full long explanation and reasoning from contextual information before colon. Explanation and the reasoning should be clear and faithful to the context!
   - After colon end with the Key.
   - Maintain third-person perspective

2. Key Requirements:
   - Key = Short Phase Key summarizing one element
   - Key Must be unique and NOT contain brackets "(", ")", or colon ":"
   - Use Title Case (e.g., "Engineering Careers")

3. Formatting Rules:
   - Strict Python list syntax with double quotes
   - Sort by importance (most important first)
   - Return empty list [] if no valid answers exist
   - Make sure only one colon ":" is used for one answer element
   
Example: ["Marie Curie was a Polish-French physicist and chemist who discovered radioactivity elements polonium/radium. She became first woman Nobel laureate (1903) and first double Nobel winner, revolutionizing radiation therapy :Marie Curie", "Alan Turing was a British mathematician who developed modern computing concepts through his Turing Machine model. He decrypted Nazi Enigma codes in WWII and established foundational AI principles in his Turing Test :Alan Turing"]
"""
        for _ in range(max_retries):
            try:
                # Call base LLM implementation
                raw_output = super()._invoke(
                    prompt=user_input,
                    system_prompt=system_prompt,
                    temperature=0,
                )

                # Extract and parse list
                list_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
                if not list_match:
                    raise ValueError("No valid list found in output")

                parsed = ast.literal_eval(list_match.group().replace("/n",""))
                if not isinstance(parsed, list):
                    raise ValueError("Output is not a list")

                # Validate each entry
                for entry in parsed:
                    if not (isinstance(entry, str) and ':' in entry):
                        raise ValueError(f"Invalid entry: {entry}")

                return parsed

            except (SyntaxError, ValueError, AttributeError, TypeError) as e:
                logging.warning(f"==============")
                logging.warning(f"Validation failed: {str(e)}")
                logging.warning(f"incorrect result: {raw_output}")
                continue

        return []

    def invoke(self, user_input: str, max_retries = None):
        if max_retries:
            pass
        else:
            max_retries = self.max_retries

        return str(self.structured_invoke(user_input, max_retries))



