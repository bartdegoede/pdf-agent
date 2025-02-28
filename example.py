#!/usr/bin/env python
"""
Example script for using the PDF Extraction Agent.
"""

import logging
import os
import time

from pdf_agent import PDFExtractionAgent

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("pdf_extraction.log"),  # Log to file
    ],
)
logger = logging.getLogger("example_script")

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("Missing OpenAI API key. Set the OPENAI_API_KEY environment variable.")
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Initialize the agent
logger.info("Initializing PDF Extraction Agent")
agent = PDFExtractionAgent(openai_api_key=api_key, openai_model="gpt-4o-mini")

# Process a PDF file
PDF_PATH = "example.pdf"  # Replace with your PDF path
if os.path.exists(PDF_PATH):
    logger.info("Starting PDF extraction for: %s", PDF_PATH)
    start_time = time.time()

    try:
        result = agent.process(PDF_PATH)

        # Get content and stats
        content = result["content"]
        stats = result["stats"]

        # Save the content to a markdown file
        OUTPUT_PATH = PDF_PATH.replace(".pdf", ".md")
        logger.info("Writing results to %s", OUTPUT_PATH)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(content)

        elapsed = time.time() - start_time
        logger.info(
            "Extraction complete in %.2f seconds! Results saved to %s",
            elapsed,
            OUTPUT_PATH,
        )

        # Display stats to the user
        print("Extraction summary:")
        print(f"- Extracted {stats['table_count']} tables")
        print(f"- Extracted {stats['image_count']} images")
        print(f"- Generated {stats['content_length']} characters of content")
        print(f"- Processed in {stats['total_time']:.2f} seconds")

        # Display token usage if available
        if "token_usage" in stats:
            token_usage = stats["token_usage"]
            print("\nToken usage:")
            print(f"- Prompt tokens: {token_usage['prompt_tokens']}")
            print(f"- Completion tokens: {token_usage['completion_tokens']}")
            print(f"- Total tokens: {token_usage['total_tokens']}")
            print(f"- API calls: {token_usage['api_calls']}")

        # Get detailed stats if available
        detailed_stats = agent.get_extraction_stats()
        if detailed_stats:
            if detailed_stats["has_tables"]:
                print(f"\nTables found on pages: {', '.join(map(str, detailed_stats['table_pages']))}")
            if detailed_stats["has_images"]:
                print(f"Images found on pages: {', '.join(map(str, detailed_stats['image_pages']))}")

        print(f"\nResults saved to {OUTPUT_PATH}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Extraction failed after %.2f seconds: %s", elapsed, str(e), exc_info=True)
        print(f"Extraction failed: {str(e)}")
else:
    logger.error("PDF file not found: %s", PDF_PATH)
    print(f"PDF file not found: {PDF_PATH}")
    print("Please update the 'pdf_path' variable with a valid PDF file path")
