import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional

import camelot
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path

# Get logger
logger = logging.getLogger("pdf_extraction_agent.table_extractor")


class TableExtractorTool:
    """Tool for extracting tables from PDFs and converting them to markdown."""

    def extract_tables(
        self, pdf_path: str, llm: Optional[Any] = None, pages: str = "all"
    ) -> List[Dict[str, Any]]:
        """Extract tables from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            llm: Vision-capable LLM for analyzing tables. If None, it will be created.
            pages: Pages to extract tables from (e.g., "1,3,4" or "all").

        Returns:
            List of extracted tables with page number and markdown.
        """
        # First try with library-based extraction
        tables = self._extract_with_camelot(pdf_path, pages)

        # If no tables are found or extraction failed, use LLM
        if not tables:
            if llm is None:
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                )
            tables = self._extract_with_llm(pdf_path, llm, pages)

        return tables

    def _extract_with_camelot(self, pdf_path: str, pages: str) -> List[Dict[str, Any]]:
        """Extract tables using Camelot."""
        logger.info(f"Extracting tables with Camelot from {pdf_path}, pages={pages}")
        start_time = time.time()
        try:
            # Convert pages parameter to format Camelot expects
            if pages == "all":
                pages = "1-end"
                logger.info("Processing all pages")
            else:
                logger.info(f"Processing specific pages: {pages}")

            # Extract tables
            logger.info("Calling Camelot to extract tables")
            extraction_start = time.time()
            tables_data = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
            extraction_time = time.time() - extraction_start
            logger.info(f"Camelot found {len(tables_data)} tables in {extraction_time:.2f} seconds")

            # Process extracted tables
            result = []
            for i, table in enumerate(tables_data):
                logger.info(f"Processing table {i+1}/{len(tables_data)}")
                table_start = time.time()
                
                # Convert to pandas DataFrame
                df = table.df

                # Get page number
                page_num = table.page
                logger.info(f"Table {i+1} is on page {page_num}")

                # Convert to markdown
                markdown = df.to_markdown(index=False)
                
                result.append(
                    {
                        "page": page_num,
                        "markdown": markdown,
                        "data": df.to_dict(orient="records"),
                    }
                )
                
                table_time = time.time() - table_start
                logger.info(f"Table {i+1} processed in {table_time:.2f} seconds")

            elapsed = time.time() - start_time
            logger.info(f"Camelot extraction completed in {elapsed:.2f} seconds, found {len(result)} tables")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error extracting tables with Camelot after {elapsed:.2f} seconds: {str(e)}", exc_info=True)
            return []

    def _extract_with_llm(
        self, pdf_path: str, llm: Any, pages: str
    ) -> List[Dict[str, Any]]:
        """Extract tables using a vision-capable LLM."""
        logger.info(f"Extracting tables with LLM from {pdf_path}, pages={pages}")
        start_time = time.time()
        try:
            # Convert PDF to images
            logger.info("Converting PDF to images for LLM table extraction")
            conversion_start = time.time()
            
            if pages == "all":
                images = convert_from_path(pdf_path)
                page_indices = list(range(len(images)))
                logger.info(f"Converting all {len(images)} pages to images")
            else:
                # Parse pages string into list of page indices (0-based)
                logger.info(f"Parsing page specification: {pages}")
                page_nums = []
                for part in pages.split(","):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        page_nums.extend(range(start, end + 1))
                    else:
                        page_nums.append(int(part))
                page_indices = [num - 1 for num in page_nums]  # Convert to 0-based
                logger.info(f"Converted to page indices (0-based): {page_indices}")
                
                images = convert_from_path(pdf_path)
                logger.info(f"PDF converted to {len(images)} total images")
                images = [images[i] for i in page_indices if i < len(images)]
                logger.info(f"Selected {len(images)} images for processing")
            
            conversion_time = time.time() - conversion_start
            logger.info(f"PDF to image conversion completed in {conversion_time:.2f} seconds")

            result = []

            for i, img in enumerate(images):
                page_num = page_indices[i] + 1  # Convert back to 1-based
                logger.info(f"Processing image {i+1}/{len(images)} (page {page_num})")
                page_start = time.time()

                # Encode image to base64 for API
                encode_start = time.time()
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                encode_time = time.time() - encode_start
                logger.info(f"Image for page {page_num} encoded in {encode_time:.2f} seconds")

                # Create prompt with the image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identify and extract all tables from this image. Convert each table to markdown format. Only include tables, not other text content. If no tables are present, respond with 'No tables found'.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}"
                                },
                            },
                        ],
                    }
                ]

                # Call LLM
                logger.info(f"Sending page {page_num} to LLM API for table extraction")
                llm_start = time.time()
                response = llm.invoke(messages)
                markdown_tables = response.content
                llm_time = time.time() - llm_start
                logger.info(f"LLM processed page {page_num} in {llm_time:.2f} seconds")

                # If tables were found
                if "No tables found" not in markdown_tables:
                    logger.info(f"Tables found on page {page_num}")
                    result.append(
                        {
                            "page": page_num,
                            "markdown": markdown_tables,
                            "data": None,  # We don't have structured data from LLM extraction
                        }
                    )
                else:
                    logger.info(f"No tables found on page {page_num}")
                
                page_time = time.time() - page_start
                logger.info(f"Completed processing page {page_num} in {page_time:.2f} seconds")

            elapsed = time.time() - start_time
            logger.info(f"LLM table extraction completed in {elapsed:.2f} seconds, found {len(result)} tables")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error extracting tables with LLM after {elapsed:.2f} seconds: {str(e)}", exc_info=True)
            return []
