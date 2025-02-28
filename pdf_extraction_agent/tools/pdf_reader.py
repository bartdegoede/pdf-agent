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

    def extract_text(
        self, pdf_path: str, llm: Optional[Any] = None, fallback_to_llm_ocr: bool = True
    ) -> str:
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
        logger.info(f"Extracting text with PyPDF from {pdf_path}")
        start_time = time.time()
        try:
            text = ""
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                logger.info(f"PDF has {len(reader.pages)} pages")
                for i, page in enumerate(reader.pages):
                    page_start = time.time()
                    logger.info(f"Extracting text from page {i+1}/{len(reader.pages)}")
                    page_text = page.extract_text()
                    page_time = time.time() - page_start
                    if page_text:
                        text += page_text + "\n\n"
                        logger.info(f"Extracted {len(page_text)} chars from page {i+1} in {page_time:.2f} seconds")
                    else:
                        logger.warning(f"No text extracted from page {i+1}")
            elapsed = time.time() - start_time
            logger.info(f"PyPDF extraction completed in {elapsed:.2f} seconds, total {len(text)} chars")
            return text
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error extracting text with PyPDF after {elapsed:.2f} seconds: {str(e)}", exc_info=True)
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
        logger.info(f"Extracting text with LLM OCR from {pdf_path}")
        start_time = time.time()
        try:
            # Convert PDF to images
            logger.info("Converting PDF to images")
            conversion_start = time.time()
            images = convert_from_path(pdf_path)
            conversion_time = time.time() - conversion_start
            logger.info(f"PDF converted to {len(images)} images in {conversion_time:.2f} seconds")
            
            all_text = ""
            total_tokens = 0

            for i, img in enumerate(images):
                logger.info(f"Processing image {i+1}/{len(images)} with LLM OCR")
                page_start = time.time()
                
                # Encode image to base64 for API
                encode_start = time.time()
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                encode_time = time.time() - encode_start
                logger.info(f"Image {i+1} encoded in {encode_time:.2f} seconds")

                # Create prompt with the image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all the text from this image. Include all text content, preserving paragraphs, bullet points, and formatting as much as possible.",
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
                logger.info(f"Sending image {i+1} to LLM API")
                llm_start = time.time()
                response = llm.invoke(messages)
                page_text = response.content
                llm_time = time.time() - llm_start
                logger.info(f"LLM returned {len(page_text)} chars for image {i+1} in {llm_time:.2f} seconds")
                
                # Check if token information is available (depends on the LLM implementation)
                if hasattr(response, 'usage') and response.usage is not None:
                    page_tokens = getattr(response.usage, 'total_tokens', 0)
                    total_tokens += page_tokens
                    logger.info(f"OCR token usage for page {i+1}: {page_tokens} tokens")

                all_text += f"Page {i+1}:\n{page_text}\n\n"
                
                page_time = time.time() - page_start
                logger.info(f"Completed processing image {i+1} in {page_time:.2f} seconds")

            total_time = time.time() - start_time
            logger.info(f"LLM OCR extraction completed in {total_time:.2f} seconds, total {len(all_text)} chars")
            if total_tokens > 0:
                logger.info(f"Total OCR token usage across all pages: {total_tokens} tokens")
            return all_text
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error extracting text with LLM OCR after {elapsed:.2f} seconds: {str(e)}", exc_info=True)
            return ""
