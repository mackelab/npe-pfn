# Contributing to TabPFN Systems

Welcome to TabPFN Systems! This repository is a collection of community-contributed packages that extend and enhance
TabPFN. We welcome all contributions and aim to make contributing as simple as possible.

## Quick Start

1. Install for development:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

2. Create your package:

```bash
mkdir src/your_package examples/your_package
```

3. Start coding!

## Repository Structure

TabPFN Systems uses a modular structure where each contribution lives in its own subpackage:

```
tabpfn_system/
├── src/
│   └── your_package/          # Your package code
├── examples/
│   └── your_package/          # Usage examples
├── tests/
│   └── your_package/          # Tests (optional)
└── requirements/
    └── your_package.txt       # Additional dependencies (if needed)
```

## Guidelines

### Package Requirements

- Python 3.8+
- TabPFN 1.0+
- Keep dependencies minimal and document them
- Avoid large data files (>10MB)
- Include license headers in source files

### Documentation

Each package should include:

- README.md with:
    - Package description
    - Basic usage example
    - Requirements
    - Author(s)
- Docstrings for public functions
- At least one example notebook/script

### Naming & Structure

- Follow standard Python package structure
- One package per contribution
- Keep code focused and modular

## Creating a Pull Request

Include in your PR description:

```markdown
## Package Description
[What does your package do?]

## Basic Usage
```python
# Short code example
```

## Dependencies

- Any additional requirements

## Checklist

- [ ] Added package under src/
- [ ] Included example(s)
- [ ] Added requirements.txt (if needed)
- [ ] Created README.md

```

## Types of Contributions

We welcome:
- Extensions for specific use cases
- Integration with other frameworks
- Utility functions
- Visualization tools
- Example applications
- Domain-specific adaptations
- Benchmarks and evaluations
- Documentation improvements

## Community Guidelines

- Be respectful and professional
- Help others learn and grow
- Give credit where due
- Follow standard open source etiquette

## Legal & Security

- Only contribute code you have rights to
- Don't include sensitive or private data
- Report security issues to security@priorlabs.ai
- All contributions must be Apache 2.0 licensed

## Example Package Structure

Here's a minimal example:

```

src/interpretability/
├── __init__.py
├── shap.py
└── utils.py

examples/interpretability/
├── basic_usage.ipynb
└── README.md

requirements/
└── interpretability.txt

```

Minimum README.md template:
```markdown
# TabPFN [Your Package Name]

## Description
Brief description of what your package does

## Installation
```bash
pip install -e ".[your_package]"
```

## Usage

```python
from tabpfn_your_package import YourClass

# Basic usage example
```

## Requirements

- List any additional dependencies

## Authors

- Your Name (@github_username)

```

## Getting Help

- Open an issue for questions
- Join our [Discord community](https://discord.gg/tabpfn)
- Email: community@priorlabs.ai
- Read our [documentation](https://docs.priorlabs.ai)

## Recognition

We believe in recognizing contributions:
- All contributors listed in CONTRIBUTORS.md
- Contributions highlighted in release notes
- Special recognition for significant contributions

Thank you for helping make TabPFN better!