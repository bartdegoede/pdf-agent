import base64
import io
import logging
import time
from typing import Any, Optional

import pypdf
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path

# Get logger
logger = logging.getLogger("pdf_extraction_agent.pdf_reader")


class PDFReaderTool:
    """Tool for extracting text from PDFs using PyPDF and Vision LLM for OCR."""

    def extract_text(self, pdf_path: str, llm: Optional[Any] = None, fallback_to_llm_ocr: bool = True) -> str:
        """Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            llm: LLM instance for OCR (must support vision). If None, it will be created.
            fallback_to_llm_ocr: Whether to use LLM-based OCR if PyPDF fails.

        Returns:
            Extracted text from the PDF.
        """
        # First try using PyPDF
        text = self._extract_with_pypdf(pdf_path)

        # If text is empty or looks incomplete and fallback is enabled, use LLM OCR
        if not text or (fallback_to_llm_ocr and self._is_text_incomplete(text)):
            if llm is None:
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                )
            text = self._extract_with_llm_ocr(pdf_path, llm)

        return text

    def _extract_with_pypdf(self, pdf_path: str) -> str:
        """Extract text using PyPDF."""
        logger.info("Extracting text with PyPDF from %s", pdf_path)
        start_time = time.time()
        try:
            text = ""
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                logger.info("PDF has %d pages", len(reader.pages))
                for i, page in enumerate(reader.pages):
                    page_start = time.time()
                    logger.info("Extracting text from page %d/%d", i + 1, len(reader.pages))
                    page_text = page.extract_text()
                    page_time = time.time() - page_start
                    if page_text:
                        text += page_text + "\n\n"
                        logger.info("Extracted %d chars from page %d in %.2f seconds", len(page_text), i + 1, page_time)
                    else:
                        logger.warning("No text extracted from page %d", i + 1)
            elapsed = time.time() - start_time
            logger.info("PyPDF extraction completed in %.2f seconds, total %d chars", elapsed, len(text))
            return text
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Error extracting text with PyPDF after %.2f seconds: %s", elapsed, str(e), exc_info=True)
            return ""

    def _is_text_incomplete(self, text: str) -> bool:
        """Check if the extracted text seems incomplete."""
        # This is a simple heuristic - improve based on your needs
        if not text:
            return True

        # If text is very short or has very few words per page, it might be incomplete
        words = text.split()
        if len(words) < 100:  # Arbitrary threshold
            return True

        return False

    def _extract_with_llm_ocr(self, pdf_path: str, llm: Any) -> str:
        """Extract text using a vision-capable LLM for OCR."""
        logger.info("Extracting text with LLM OCR from %s", pdf_path)
        start_time = time.time()
        try:
            # Convert PDF to images
            logger.info("Converting PDF to images")
            conversion_start = time.time()
            images = convert_from_path(pdf_path)
            conversion_time = time.time() - conversion_start
            logger.info("PDF converted to %d images in %.2f seconds", len(images), conversion_time)

            all_text = ""
            total_tokens = 0

            for i, img in enumerate(images):
                logger.info("Processing image %d/%d with LLM OCR", i + 1, len(images))
                page_start = time.time()

                # Encode image to base64 for API
                encode_start = time.time()
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                encode_time = time.time() - encode_start
                logger.info("Image %d encoded in %.2f seconds", i + 1, encode_time)

                # Create prompt with the image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all the text from this image. "
                                "Include all text content, preserving paragraphs, "
                                "bullet points, and formatting as much as possible.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_str}"},
                            },
                        ],
                    }
                ]

                # Call LLM
                logger.info("Sending image %d to LLM API", i + 1)
                llm_start = time.time()
                response = llm.invoke(messages)
                page_text = response.content
                llm_time = time.time() - llm_start
                logger.info("LLM returned %d chars for image %d in %.2f seconds", len(page_text), i + 1, llm_time)

                # Check if token information is available (depends on the LLM implementation)
                if hasattr(response, "usage") and response.usage is not None:
                    page_tokens = getattr(response.usage, "total_tokens", 0)
                    total_tokens += page_tokens
                    logger.info("OCR token usage for page %d: %d tokens", i + 1, page_tokens)

                all_text += f"Page {i + 1}:\n{page_text}\n\n"

                page_time = time.time() - page_start
                logger.info("Completed processing image %d in %.2f seconds", i + 1, page_time)

            total_time = time.time() - start_time
            logger.info("LLM OCR extraction completed in %.2f seconds, total %d chars", total_time, len(all_text))
            if total_tokens > 0:
                logger.info("Total OCR token usage across all pages: %d tokens", total_tokens)
            return all_text
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Error extracting text with LLM OCR after %.2f seconds: %s", elapsed, str(e), exc_info=True)
            return ""
