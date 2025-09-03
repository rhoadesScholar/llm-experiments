# {{project_name}}

## {{project_description}}

![GitHub - License](https://img.shields.io/github/license/{{github_username}}/{{pypi_package_name}})
[![CI/CD Pipeline](https://github.com/{{github_username}}/{{repo_name}}/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/{{github_username}}/{{repo_name}}/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/github/{{github_username}}/{{repo_name}}/graph/badge.svg?token={{codecov_token}})](https://codecov.io/github/{{github_username}}/{{repo_name}})
![PyPI - Version](https://img.shields.io/pypi/v/{{pypi_package_name}})
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/{{pypi_package_name}})

{{long_description}}

## Installation

### From PyPI

```bash
pip install {{pypi_package_name}}
```

### From source

```bash
pip install git+https://github.com/{{github_username}}/{{repo_name}}.git
```

## Usage

```python
import {{package_name}}

# Example usage
# TODO: Add your usage examples here
```

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/{{github_username}}/{{repo_name}}.git
cd {{repo_name}}

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

{{license_name}}. See [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite it using the information in [CITATION.cff](CITATION.cff).
