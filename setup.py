from setuptools import find_packages, setup

setup(
    name="pdf-extraction-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.19",
        "langgraph>=0.3.1",
        "openai>=1.65.1",
        "pypdf>=4.0.2",
        "pillow>=10.2.0",
        "pydantic>=2.10.6",
        "pandas>=2.2.0",
        "pdf2image>=1.17.0",
        "camelot-py[cv]>=0.11.0",
        "click>=8.1.7",
        "tabulate>=0.9.0",
    ],
    entry_points={
        "console_scripts": [
            "pdf-extract=cli:main",
        ],
    },
    python_requires=">=3.10",
)
