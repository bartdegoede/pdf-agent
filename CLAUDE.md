# PDF Extraction Agent Guidelines

## Build/Test Commands
- Setup: `pip install -e .`
- Run tests: `pytest`
- Run single test: `pytest tests/test_file.py::test_function`
- Lint: `ruff check .`
- Type check: `mypy .`

## Code Style Guidelines
- **Formatting**: Use Black with default settings
- **Imports**: Group standard library, third-party, local imports with a blank line between groups
- **Types**: Use type hints for all function parameters and return values
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Error handling**: Use specific exceptions with contextual error messages
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Configuration**: Use environment variables for configuration parameters

Remember to write unit tests for all new functionality.