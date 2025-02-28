import base64
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path
from PIL import Image

# Get logger
logger = logging.getLogger("pdf_extraction_agent.image_extractor")


class ImageExtractorTool:
    """Tool for extracting images from PDFs with AI-generated descriptions."""

    def extract_images(
        self,
        pdf_path: str,
        llm: Optional[Any] = None,
        save_images: bool = False,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract images from a PDF file with AI-generated descriptions.

        Args:
            pdf_path: Path to the PDF file.
            llm: Vision-capable LLM for image descriptions. If None, it will be created.
            save_images: Whether to save extracted images to disk.
            output_dir: Directory to save images to. If None, a temporary directory is used.

        Returns:
            List of extracted images with page number, filename, and description.
        """
        if llm is None:
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
            )

        # Create output directory if needed
        if save_images:
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(pdf_path), "extracted_images")
            os.makedirs(output_dir, exist_ok=True)

        # Extract images from PDF
        images = self._extract_images_from_pdf(pdf_path)

        # Get descriptions for images
        result = []
        for i, img_data in enumerate(images):
            # Generate description using LLM
            description = self._generate_description(img_data["image"], llm)

            # Save image if requested
            filename = None
            if save_images:
                filename = f"page_{img_data['page']}_image_{i+1}.png"
                img_path = os.path.join(output_dir, filename)
                img_data["image"].save(img_path)

            result.append(
                {
                    "page": img_data["page"],
                    "filename": filename,
                    "description": description,
                    "image": img_data["image"] if not save_images else None,
                }
            )

        return result

    def _extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF pages."""
        logger.info(f"Extracting images from PDF: {pdf_path}")
        start_time = time.time()
        try:
            # Convert PDF pages to images
            logger.info("Converting PDF pages to images")
            conversion_start = time.time()
            page_images = convert_from_path(pdf_path)
            conversion_time = time.time() - conversion_start
            logger.info(f"PDF converted to {len(page_images)} images in {conversion_time:.2f} seconds")
            
            result = []

            for page_num, page_image in enumerate(page_images, 1):
                logger.info(f"Processing page {page_num}/{len(page_images)} for image extraction")
                # For real image extraction, we would use more sophisticated methods
                # such as object detection to identify image regions
                # For now, we'll use the whole page as an image
                result.append({"page": page_num, "image": page_image})

            elapsed = time.time() - start_time
            logger.info(f"Image extraction completed in {elapsed:.2f} seconds, extracted {len(result)} images")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error extracting images from PDF after {elapsed:.2f} seconds: {str(e)}", exc_info=True)
            return []

    def _generate_description(self, image: Image.Image, llm: Any) -> str:
        """Generate a description for an image using a vision-capable LLM."""
        start_time = time.time()
        try:
            # Encode image to base64 for API
            logger.info("Encoding image for LLM description")
            encode_start = time.time()
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            encode_time = time.time() - encode_start
            logger.info(f"Image encoded in {encode_time:.2f} seconds")

            # Create prompt with the image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Focus on the visual content, any text present, and its relevance to the document. Provide a comprehensive description that could replace the image if needed.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"},
                        },
                    ],
                }
            ]

            # Call LLM
            logger.info("Sending image to LLM API for description")
            llm_start = time.time()
            response = llm.invoke(messages)
            description = response.content
            llm_time = time.time() - llm_start
            logger.info(f"LLM generated description ({len(description)} chars) in {llm_time:.2f} seconds")
            
            # Note: For the vision API, we can't directly count tokens as easily as with text
            # Token usage may be tracked in the llm object if available
            if hasattr(response, 'usage') and response.usage is not None:
                logger.info(f"Image token usage - Total: {getattr(response.usage, 'total_tokens', 'unknown')}")
            
            elapsed = time.time() - start_time
            logger.info(f"Description generation completed in {elapsed:.2f} seconds")
            return description
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error generating image description after {elapsed:.2f} seconds: {str(e)}", exc_info=True)
            return "Image description unavailable"
