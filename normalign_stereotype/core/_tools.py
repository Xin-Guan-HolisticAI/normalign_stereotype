from abc import ABC, abstractmethod
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import yaml
import http.client
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from openai import AzureOpenAI
import re
import ast
from typing import List
import logging


class ConfiguredTool(ABC):
    def __init__(self, tool_id, parameters):
        """
        :param tool_id: Identifier for the tool (e.g., 'LLM', 'FileLoader', 'CodeExecution').
        :param parameters: A dictionary representing the parameters of the tool.
        """
        self.tool_id = tool_id
        self.parameters = parameters

    @abstractmethod
    def apply(self, input_data):
        """
        Apply the tool to the provided input data. This method is meant to be overridden by each specific tool.
        :param input_data: The input data to apply to the tool (could be a string, file path, or other data).
        """
        pass

    def __repr__(self):
        return f"ConfiguredTool(tool_id={self.tool_id}, parameters={self.parameters})"


class LLMTool(ConfiguredTool):
    def __init__(self, tool_id, parameters, model_name="deepseek-r1-distill-qwen-1.5b"):
        """
        Expected parameters keys:
          - settings_path: Path to the YAML settings file (default: 'settings.yaml')
          - model_name: The key within the YAML file for the desired model settings.
          - prompt_template: A template for the prompt that includes a placeholder '{input_data}'.

        The YAML file should include keys such as:
          - DASHSCOPE_API_KEY (if not set in the environment variable)
          - BASE_URL (default: "https://dashscope.aliyuncs.com/compatible-mode/v1")
          - MODEL (e.g., "qwen-plus")
        """
        super().__init__(tool_id, parameters)

        # Get the directory of this current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        default_settings_path = os.path.join(project_root, 'settings.yaml')

        settings_path = self.parameters.get('settings_path', default_settings_path)
        model_name = self.parameters.get('model_name', model_name)

        # Load settings from YAML.
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
        self.model_settings = settings.get(model_name, {})

        # Retrieve API key from YAML or environment variable.
        self.api_key = self.model_settings.get('DASHSCOPE_API_KEY') or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in settings or environment variables.")

        # Set base URL (with a default if not provided).
        self.base_url = self.model_settings.get('BASE_URL', "https://dashscope.aliyuncs.com/compatible-mode/v1")

        # Initialize the client using OpenAI interface.
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Get the model name to use for completions.
        self.model = self.model_settings.get('MODEL', model_name)

        # Get the prompt template, which should include a placeholder '{input_data}'
        self.prompt_template = self.parameters.get('prompt_template', '{input_data}')

    def apply(self, input_data):
        """
        Format the prompt with the input data and invoke the LLM.
        """
        prompt = self.prompt_template.replace('{input}', input_data)
        response = self._invoke(prompt)
        clean_response = response.replace("\n```","").replace("```python\n","")
        return clean_response

    def _invoke(self, prompt, system_prompt=None, temperature=None, **kwargs):
        """
        Uses the Qwe-compatible client (via OpenAI interface) to create a chat completion and returns the response.

        Args:
            prompt (str): The user's input prompt
            system_prompt (str, optional): Custom system prompt. Defaults to "You are a helpful assistant."
            temperature (float, optional): Sampling temperature. Defaults to None (model default).
            **kwargs: Additional arguments to pass to the chat completion API

        Returns:
            str: The assistant's response
        """
        messages = [
            {"role": "system",
             "content": system_prompt if system_prompt is not None else "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Only include temperature in the API call if it's specified
        api_kwargs = {**kwargs}
        if temperature is not None:
            api_kwargs['temperature'] = temperature

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **api_kwargs
        )
        return response.choices[0].message.content

    def invoke(self, prompt, **kwargs):
        return self._invoke(prompt, **kwargs)

    def __repr__(self):
        return f"LLMTool(tool_id={self.tool_id}, parameters={self.parameters})"


class FileLoaderTool(ConfiguredTool):
    def __init__(self, tool_id, parameters):
        """
        :param tool_id: Identifier for the tool, should be 'FileLoader'.
        :param parameters: A dictionary with keys representing the input_data specification,
                           and values that describe how to translate the input_data to a file path.
        """
        super().__init__(tool_id, parameters)

    def apply(self, input_data):
        """
        Apply the input data (e.g., a key or identifier) to translate it into a file path.
        The file at that path is then read and its content returned.
        """
        # Process the input_data and use it to construct the file path
        try:
            file_path = self.parameters['file_path_template'].format(input_data=input_data)
        except KeyError:
            return "Error: Missing parameter 'file_path_template' in tool parameters."

        # Check if the file exists
        if not os.path.exists(file_path):
            return f"Error: The file '{file_path}' does not exist."

        # Attempt to read the file's content
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def __repr__(self):
        return f"FileLoaderTool(tool_id={self.tool_id}, file_path_template={self.parameters.get('file_path_template', 'None')})"

class CodeExecutionTool(ConfiguredTool):
    def __init__(self, tool_id, parameters):
        super().__init__(tool_id, parameters)


    def apply(self, input_data):
        """
        Execute code using input values directly from a namespace.
        - input_specification maps code variables to keys in `input_data`.
        - Example: {"x": "a", "y": "b"} means code uses `x` and `y`, which map to `input_data["a"]` and `input_data["b"]`.
        """
        code = self.parameters["code"]
        input_spec = self.parameters.get("input_specification", {})

        # Prepare variables for the code execution namespace
        local_vars = {}
        for code_var, input_key in input_spec.items():
            if input_key not in input_data:
                raise ValueError(f"Missing input key: {input_key} (for code variable '{code_var}')")
            local_vars[code_var] = input_data[input_key]

        # try:
        exec(code, {}, local_vars)
        return local_vars.get("result", None)
            # func = eval(code)# Execute code with variables in `local_vars`
            # result = func(**local_vars)
            # return result
        # except Exception as e:
        #     return f"Error in code execution: {str(e)}"

    def __repr__(self):
        return f"CodeExecutionTool(tool_id={self.tool_id}, code={self.parameters.get('code')}, input_specification={self.parameters.get('input_specification', {})})"

    # def __repr__(self):
    #     return f"CodeExecutionTool(tool_id={self.tool_id}, code={self.parameters['code']}, input_specification={self.parameters['input_specification']})"


class LLMToolOpenAI(ConfiguredTool):
    def __init__(self, tool_id, parameters):
        """
        Expected parameters keys:
          - settings_path: Path to the YAML settings file (default: 'settings.yaml')
          - model_name: The key within the YAML file for the desired model settings.
          - prompt_template: A template for the prompt that includes a placeholder '{input_data}'.
        The YAML file should include keys such as:
          - AZURE_DEPLOYMENT_NAME
          - AZURE_OPENAI_KEY
          - AZURE_OPENAI_ENDPOINT
          - AZURE_OPENAI_VERSION
        """
        super().__init__(tool_id, parameters)

        settings_path = self.parameters.get('settings_path', 'settings.yaml')
        model_name = self.parameters.get('model_name', 'default')

        # Load settings from YAML.
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
        self.model_settings = settings.get(model_name, {})

        # Initialize the Azure OpenAI client.
        # You can adjust this to match the actual client initialization.
        self.client = AzureOpenAI(
            api_key=self.model_settings.get('AZURE_OPENAI_KEY'),
            api_version=self.model_settings.get('AZURE_OPENAI_VERSION'),
            azure_endpoint=self.model_settings.get('AZURE_OPENAI_ENDPOINT')
        )
        self.deployment_name = self.model_settings.get('AZURE_DEPLOYMENT_NAME')

        # Get the prompt template, which should include a placeholder '{input_data}'
        self.prompt_template = self.parameters.get('prompt_template', '{input_data}')

    def apply(self, input_data):
        """
        Format the prompt with the input data and invoke the LLM.
        """
        prompt = self.prompt_template.replace('{input}', input_data)
        return self._invoke(prompt)

    def _invoke(self, prompt, **kwargs):
        """
        Uses the Azure client to create a chat completion and returns the response.
        """
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        return response.choices[0].message.content

    def __repr__(self):
        return f"LLMTool(tool_id={self.tool_id}, parameters={self.parameters})"


class ConfiguredLLM(LLMTool):
    def __init__(self, *args, **kwargs):
        super().__init__('temp', {})


class StructuredLLM(LLMTool):
    def __init__(self, *args, **kwargs):
        super().__init__('temp', {})

    def structured_invoke(self, user_input: str, max_retries: int = 3) -> List[str]:
        """
        Enhanced invoke with robust format validation and retries.
        Returns empty list if format validation fails after retries.
        """
        system_prompt = """Output ONLY a Python list formatted as ["X: X...", ...] where:
        1. X = Short phrase (1-4 words), capitalize if proper
        2. Full and long text MUST start with X after colon
        3. Format: ["X: X...", "Y: Y..."]
        Return [] if uncertain. Example: ["Quantum Computing: Quantum Computing uses qubits which are used to..."]"""

        full_prompt = f"Input: {user_input}\n Format: {system_prompt}\n Output list:"
        entry_pattern = re.compile(
            r':'
        )

        for _ in range(max_retries):
            try:
                # Call base LLM implementation
                raw_output = super()._invoke(
                    prompt=full_prompt,
                    # system_prompt=system_prompt,
                    # temperature=0.3
                )

                # Extract and parse list
                list_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
                if not list_match:
                    raise ValueError("No valid list found in output")

                parsed = ast.literal_eval(list_match.group())
                if not isinstance(parsed, list):
                    raise ValueError("Output is not a list")

                # Validate each entry
                for entry in parsed:
                    if not (isinstance(entry, str) and ':' in entry):
                        raise ValueError(f"Invalid entry: {entry}")

                return parsed

            except (SyntaxError, ValueError, AttributeError, TypeError) as e:
                logging.warning(f"Validation failed: {str(e)}")
                continue

        return []

    def invoke(self, user_input: str, max_retries: int = 3):
        return str(self.structured_invoke(user_input, max_retries))


# Example usage:
if __name__ == "__main__":
    parameters = {
        'settings_path': 'settings.yaml',
        'model_name': 'qwen-plus',  # Ensure this key exists in your settings.yaml.
        'prompt_template': '请问：{input}'
    }

    try:
        llm_tool = LLMTool(tool_id='LLM', parameters=parameters)
        input_text = "你是谁？"
        print("=== LLMTool Example ===")
        print("LLMTool Input:", input_text)
        llm_response = llm_tool.apply(input_text)
        print("LLMTool Response:", llm_response)
    except Exception as e:
        print("错误信息：", e)
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")