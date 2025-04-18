import yaml
import os
from openai import OpenAI
from constants import CURRENT_DIR, PROJECT_ROOT


class LLMFactory:
    """
    Factory class to create LLM instances based on model name and configuration
    """

    def __init__(self, model_name, settings_path=os.path.join(PROJECT_ROOT, "settings.yaml")):
        """
        Initialize LLM with specified model from settings

        Args:
            model_name (str): Name of the model to use
            settings_path (str): Path to the settings YAML file
        """
        self.model_name = model_name

        # Load settings
        with open(settings_path, 'r') as f:
            self.settings = yaml.safe_load(f)

        # Validate model exists in settings
        if model_name not in self.settings:
            raise ValueError(f"Model '{model_name}' not found in settings")

        # Get API key for the specified model
        self.api_key = self.settings[model_name].get('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError(f"API key not found for model '{model_name}'")

        self.base_url = self.settings.get('BASE_URL', "https://dashscope.aliyuncs.com/compatible-mode/v1")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def run_prompt(self, prompt_template_name, **kwargs):
        """
        Run a prompt through the LLM using a template

        Args:
            prompt_template_name (str): name of the prompt template file
            **kwargs: Variables to substitute in the template

        Returns:
            str: The LLM response
        """
        # Load prompt template
        prompt_template_path = os.path.join(CURRENT_DIR, "prompts", f"{prompt_template_name}.txt")
        with open(prompt_template_path, 'r') as f:
            template = f.read()

        # Replace variables in template
        prompt = template.format(**kwargs)

        # Run the prompt through the LLM
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "You are a linguistic analysis system."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content