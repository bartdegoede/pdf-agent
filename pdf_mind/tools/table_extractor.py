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

    def extract_tables(self, pdf_path: str, llm: Optional[Any] = None, pages: str = "all") -> List[Dict[str, Any]]:
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
        logger.info("Extracting tables with Camelot from %s, pages=%s", pdf_path, pages)
        start_time = time.time()
        try:
            # Convert pages parameter to format Camelot expects
            if pages == "all":
                pages = "1-end"
                logger.info("Processing all pages")
            else:
                logger.info("Processing specific pages: %s", pages)

            # Extract tables
            logger.info("Calling Camelot to extract tables")
            extraction_start = time.time()
            tables_data = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
            extraction_time = time.time() - extraction_start
            logger.info("Camelot found %d tables in %0.2f seconds", len(tables_data), extraction_time)

            # Process extracted tables
            result = []
            for i, table in enumerate(tables_data):
                logger.info("Processing table %d/%d", i + 1, len(tables_data))
                table_start = time.time()

                # Convert to pandas DataFrame
                df = table.df

                # Get page number
                page_num = table.page
                logger.info("Table %d is on page %d", i + 1, page_num)

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
                logger.info("Table %d processed in %0.2f seconds", i + 1, table_time)

            elapsed = time.time() - start_time
            logger.info("Camelot extraction completed in %0.2f seconds, found %d tables", elapsed, len(result))
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Error extracting tables with Camelot after %0.2f seconds: %s", elapsed, str(e), exc_info=True)
            return []

    def _extract_with_llm(self, pdf_path: str, llm: Any, pages: str) -> List[Dict[str, Any]]:
        """Extract tables using a vision-capable LLM."""
        logger.info("Extracting tables with LLM from %s, pages=%s", pdf_path, pages)
        start_time = time.time()
        try:
            # Convert PDF to images
            logger.info("Converting PDF to images for LLM table extraction")
            conversion_start = time.time()

            if pages == "all":
                images = convert_from_path(pdf_path)
                page_indices = list(range(len(images)))
                logger.info("Converting all %d pages to images", len(images))
            else:
                # Parse pages string into list of page indices (0-based)
                logger.info("Parsing page specification: %s", pages)
                page_nums: List[int] = []
                for part in pages.split(","):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        page_nums.extend(range(start, end + 1))
                    else:
                        page_nums.append(int(part))
                page_indices = [num - 1 for num in page_nums]  # Convert to 0-based
                logger.info("Converted to page indices (0-based): %s", page_indices)

                images = convert_from_path(pdf_path)
                logger.info("PDF converted to %d total images", len(images))
                images = [images[i] for i in page_indices if i < len(images)]
                logger.info("Selected %d images for processing", len(images))

            conversion_time = time.time() - conversion_start
            logger.info("PDF to image conversion completed in %0.2f seconds", conversion_time)

            result = []

            for i, img in enumerate(images):
                page_num = page_indices[i] + 1  # Convert back to 1-based
                logger.info("Processing image %d/%d (page %d)", i + 1, len(images), page_num)
                page_start = time.time()

                # Encode image to base64 for API
                encode_start = time.time()
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                encode_time = time.time() - encode_start
                logger.info("Image for page %d encoded in %0.2f seconds", page_num, encode_time)

                # Create prompt with the image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identify and extract all tables from this image. "
                                "Convert each table to markdown format. "
                                "Only include tables, not other text content. "
                                "If no tables are present, respond with 'No tables found'.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_str}"},
                            },
                        ],
                    }
                ]

                # Call LLM
                logger.info("Sending page %d to LLM API for table extraction", page_num)
                llm_start = time.time()
                response = llm.invoke(messages)
                markdown_tables = response.content
                llm_time = time.time() - llm_start
                logger.info("LLM processed page %d in %0.2f seconds", page_num, llm_time)

                # Check if token information is available
                if hasattr(response, "usage") and response.usage is not None:
                    page_tokens = getattr(response.usage, "total_tokens", 0)
                    logger.info("Table extraction token usage for page %d: %d tokens", page_num, page_tokens)

                # If tables were found
                if "No tables found" not in markdown_tables:
                    logger.info("Tables found on page %d", page_num)
                    result.append(
                        {
                            "page": page_num,
                            "markdown": markdown_tables,
                            "data": None,  # We don't have structured data from LLM extraction
                        }
                    )
                else:
                    logger.info("No tables found on page %d", page_num)

                page_time = time.time() - page_start
                logger.info("Completed processing page %d in %0.2f seconds", page_num, page_time)

            elapsed = time.time() - start_time
            logger.info("LLM table extraction completed in %0.2f seconds, found %d tables", elapsed, len(result))
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Error extracting tables with LLM after %0.2f seconds: %s", elapsed, str(e), exc_info=True)
            return []
