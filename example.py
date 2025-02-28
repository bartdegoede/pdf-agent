#!/usr/bin/env python
"""
Example script for using the PDF Extraction Agent.
"""

import os
import logging
import time

from pdf_extraction_agent import PDFExtractionAgent

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("pdf_extraction.log")  # Log to file
    ]
)
logger = logging.getLogger("example_script")

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("Missing OpenAI API key. Set the OPENAI_API_KEY environment variable.")
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Initialize the agent
logger.info("Initializing PDF Extraction Agent")
agent = PDFExtractionAgent(openai_api_key=api_key)

# Process a PDF file
PDF_PATH = "example.pdf"  # Replace with your PDF path
if os.path.exists(PDF_PATH):
    logger.info(f"Starting PDF extraction for: {PDF_PATH}")
    start_time = time.time()
    
    try:
        result = agent.process(PDF_PATH)
        
        # Save the result to a markdown file
        OUTPUT_PATH = PDF_PATH.replace(".pdf", ".md")
        logger.info(f"Writing results to {OUTPUT_PATH}")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(result)
        
        elapsed = time.time() - start_time
        logger.info(f"Extraction complete in {elapsed:.2f} seconds! Results saved to {OUTPUT_PATH}")
        print(f"Extraction complete! Results saved to {OUTPUT_PATH}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Extraction failed after {elapsed:.2f} seconds: {str(e)}", exc_info=True)
        print(f"Extraction failed: {str(e)}")
else:
    logger.error(f"PDF file not found: {PDF_PATH}")
    print(f"PDF file not found: {PDF_PATH}")
    print("Please update the 'pdf_path' variable with a valid PDF file path")
