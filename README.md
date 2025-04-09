# Stereotype Analysis with DOT Graphs

This project provides tools for analyzing stereotypes in text using DOT graph-based inference models. The system processes statements through a series of analysis steps to identify and classify stereotypes.

## Project Structure

### process_dot/
- `plan_with_dot.py`: Main implementation for parsing DOT files and executing inference plans
- `example_stereotype.dot`: Example DOT file showing basic stereotype analysis structure
- `example_stereotype_corrected.dot`: Corrected version with proper view accumulation
- `stereotype_graphvis_draft.dot`: Complex stereotype analysis graph structure

## Features

- DOT graph-based inference model
- View accumulation through perception paths
- Classification nodes for stereotype analysis
- Proper handling of perception and actuation edges
- Memory management for inference execution

## Usage

The system uses DOT files to define the inference structure. Each node in the graph represents a concept or classification, and edges represent either perception or actuation relationships.

Example:
```dot
statements [xlabel="{'statements'}"];
stereotype_detection [xlabel="{'statements', 'stereotype_detection'}"];
stereotype_detection_classification [xlabel="{'stereotype_detection_classification'}"];

statements -> stereotype_detection [label="perc"];
stereotype_detection_classification -> stereotype_detection [label="actu"];
```

## Requirements

- Python 3.x
- Required Python packages (see requirements.txt)
- DOT graph processing capabilities

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
