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
pdf_path = "path/to/your/document.pdf"  # Replace with your PDF path
if os.path.exists(pdf_path):
    print(f"Processing {pdf_path}...")
    result = agent.process(pdf_path)
    
    # Save the result to a markdown file
    output_path = pdf_path.replace(".pdf", ".md")
    with open(output_path, "w") as f:
        f.write(result)
    
    print(f"Extraction complete! Results saved to {output_path}")
else:
    print(f"PDF file not found: {pdf_path}")
    print("Please update the 'pdf_path' variable with a valid PDF file path")