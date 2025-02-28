#!/usr/bin/env python
"""
Example script for using the PDF Extraction Agent.
"""

import os

from pdf_extraction_agent import PDFExtractionAgent

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Initialize the agent
agent = PDFExtractionAgent(openai_api_key=api_key)

# Process a PDF file
PDF_PATH = "example.pdf"  # Replace with your PDF path
if os.path.exists(PDF_PATH):
    print(f"Processing {PDF_PATH}...")
    result = agent.process(PDF_PATH)

    # Save the result to a markdown file
    OUTPUT_PATH = PDF_PATH.replace(".pdf", ".md")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"Extraction complete! Results saved to {OUTPUT_PATH}")
else:
    print(f"PDF file not found: {PDF_PATH}")
    print("Please update the 'pdf_path' variable with a valid PDF file path")
