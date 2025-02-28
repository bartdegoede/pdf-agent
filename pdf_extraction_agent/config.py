import os
from typing import Optional

from pydantic import BaseModel, Field


class PDFExtractionConfig(BaseModel):
    """Configuration for PDF extraction agent."""

    # OpenAI configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key. If not provided, will try to read from OPENAI_API_KEY env var.",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for text generation and vision tasks.",
    )

    # PDF processing options
    use_llm_ocr: bool = Field(
        default=True,
        description="Whether to use LLM-based OCR when PyPDF extraction fails.",
    )
    extract_tables: bool = Field(
        default=True, description="Whether to extract tables from the PDF."
    )
    extract_images: bool = Field(
        default=True, description="Whether to extract images from the PDF."
    )
    save_images: bool = Field(
        default=False, description="Whether to save extracted images to disk."
    )
    output_dir: Optional[str] = Field(
        default=None, description="Directory to save extracted images to."
    )

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from config or environment variable."""
        if self.openai_api_key:
            return self.openai_api_key

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it in the config or "
                "set the OPENAI_API_KEY environment variable."
            )

        return api_key
