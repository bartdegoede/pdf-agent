# PDFMind

An agent for extracting structured content from PDFs using LangGraph and OpenAI.

## Features

- Extract and format text content from PDFs
- Convert tables to markdown format
- Extract images with AI-generated descriptions
- Use LangGraph for agent-based orchestration

## Setup

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
# Or install from pypi
pip install pdf_mind

# Install other dependencies
brew install ghostscript
brew install poppler
# apt install ghostscript poppler
```

## Usage

```python
from pdf_mind import PDFExtractionAgent

agent = PDFExtractionAgent()
result = agent.process("path/to/document.pdf")
print(result)
```

Alternatively, look at [example.py](example.py) for an example that will output metadata on extracted items and token usage:

## Development

```bash
# Run tests
poetry run pytest

# Lint code
poetry run ruff check .
poetry run black .
```
