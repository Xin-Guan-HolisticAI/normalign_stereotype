from normalign_stereotype.core._config import PROJECT_ROOT

import json
import os

class PromptManager:
    def __init__(self):
        self.script_dir = os.path.join(PROJECT_ROOT, "normalign_stereotype", "prompt_manager", "prompt_templates")

    def _load_prompts(self, kind, template_name):
        """Load prompts from the JSON file. Create default if doesn't exist."""
        file_path = os.path.join(self.script_dir, kind, template_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt = json.load(f)
                return prompt
        except FileNotFoundError:
            raise ValueError(f"Template file not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in template file: {file_path}")



    # def _create_default_prompts(self):
    #     """Create a default prompts file if it doesn't exist."""
    #     default_prompts = {
    #         "classification": {
    #             "template": "Classify the following text: {text}",
    #             "parameters": ["text"]
    #         },
    #         "generation": {
    #             "template": "Generate content about: {topic}",
    #             "parameters": ["topic"]
    #         }
    #     }
    #     self._save_prompts(default_prompts)
    #
    # def _save_prompts(self, prompts=None):
    #     """Save prompts to the JSON file."""
    #     prompts = prompts or self.prompts
    #     os.makedirs(os.path.dirname(self.prompts_file), exist_ok=True)
    #     with open(self.prompts_file, 'w', encoding='utf-8') as f:
    #         json.dump(prompts, f, indent=2)
    #
    # def get_prompt(self, prompt_name):
    #     """Get a prompt by name."""
    #     if prompt_name not in self.prompts:
    #         raise KeyError(f"Prompt '{prompt_name}' not found")
    #     return self.prompts[prompt_name]
    #
    # def add_prompt(self, name, template, parameters=None):
    #     """Add a new prompt template."""
    #     self.prompts[name] = {
    #         "template": template,
    #         "parameters": parameters or []
    #     }
    #     self._save_prompts()
    #
    # def update_prompt(self, name, template=None, parameters=None):
    #     """Update an existing prompt template."""
    #     if name not in self.prompts:
    #         raise KeyError(f"Prompt '{name}' not found")
    #
    #     if template:
    #         self.prompts[name]["template"] = template
    #     if parameters:
    #         self.prompts[name]["parameters"] = parameters
    #
    #     self._save_prompts()
    #
    # def format_prompt(self, prompt_name, **kwargs):
    #     """Format a prompt with provided parameters."""
    #     prompt = self.get_prompt(prompt_name)
    #     missing_params = set(prompt["parameters"]) - set(kwargs.keys())
    #     if missing_params:
    #         raise ValueError(f"Missing parameters: {missing_params}")
    #     return prompt["template"].format(**kwargs)
    #
    # def list_prompts(self):
    #     """List all available prompts."""
    #     return list(self.prompts.keys())
    #
    # def remove_prompt(self, prompt_name):
    #     """Remove a prompt template."""
    #     if prompt_name not in self.prompts:
    #         raise KeyError(f"Prompt '{prompt_name}' not found")
    #     del self.prompts[prompt_name]
    #     self._save_prompts()