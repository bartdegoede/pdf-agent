import base64
import io
from typing import Any, Optional

import pypdf
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path


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
        try:
            text = ""
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text
        except Exception as e:
            print(f"Error extracting text with PyPDF: {e}")
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
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            all_text = ""

            for i, img in enumerate(images):
                # Encode image to base64 for API
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

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
                response = llm.invoke(messages)
                page_text = response.content

                all_text += f"Page {i+1}:\n{page_text}\n\n"

            return all_text
        except Exception as e:
            print(f"Error extracting text with LLM OCR: {e}")
            return ""
