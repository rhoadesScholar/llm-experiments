# {{project_name}}

## {{project_description}}

![GitHub - License](https://img.shields.io/github/license/rhoadesScholar/llm-experiments)
[![CI/CD Pipeline](https://github.com/rhoadesScholar/llm-experiments/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rhoadesScholar/llm-experiments/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/github/rhoadesScholar/llm-experiments/graph/badge.svg?token=)](https://codecov.io/github/rhoadesScholar/llm-experiments)
![PyPI - Version](https://img.shields.io/pypi/v/llm-experiments)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llm-experiments)

A Python package for a repository of experiments inspecting llm's, particularly focused on detecting signs of human-like self-awareness via psychodynamic-inspired prompting..

## Installation

### From PyPI

```bash
pip install llm-experiments
```

### From source

```bash
pip install git+https://github.com/rhoadesScholar/llm-experiments.git
```

## Usage

```python
import llm_experiments

# Example usage
# TODO: Add your usage examples here
```

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/rhoadesScholar/llm-experiments.git
cd llm-experiments

# Install in development mode with all dependencies
make dev-setup
```

### Running tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run fast tests (stop on first failure)
make test-fast
```

### Code quality

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check

# Run all checks
make check-all
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite (`make test`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

GNU GPL v3.0. See [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite it using the information in [CITATION.cff](CITATION.cff).
