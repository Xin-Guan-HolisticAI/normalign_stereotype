# Normalign Stereotype

A Python-based framework for conceptual inference and stereotype analysis using advanced language models.

## Overview

This project provides tools and frameworks for:
- Conceptual inference and analysis
- Stereotype detection and analysis
- Advanced language model integration
- Reference-based data structures
- Memory management for AI agents

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Xin-Guan-HolisticAI/normalign_stereotype.git
cd normalign_stereotype
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `conceptual_inference/`: Core framework for conceptual inference
  - `_agent.py`: Agent implementation
  - `_concept.py`: Concept handling
  - `_inference.py`: Inference engine
  - `_reference.py`: Reference data structures
  - `_tools.py`: Utility functions
- `stereotype_concepts/`: Templates and definitions for stereotype analysis
- `test_results/`: Results from model testing

## Usage

1. Set up your environment variables in `settings.yaml` (not tracked in git)
2. Import and use the framework:
```python
from conceptual_inference import Agent, Concept, Inference
```

## Memory Management

The framework uses JSON-based memory files for persistence:
- `memory.json`: Main memory file
- `working_memory.json`: Temporary working memory
- `memory_*.json`: Specialized memory files

These files are excluded from version control for security and privacy.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]

## Contact

[Add contact information here]
