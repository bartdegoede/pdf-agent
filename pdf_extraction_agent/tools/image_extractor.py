import base64
import io
import os
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path
from PIL import Image


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
        try:
            # Convert PDF pages to images
            page_images = convert_from_path(pdf_path)
            result = []

            for page_num, page_image in enumerate(page_images, 1):
                # For real image extraction, we would use more sophisticated methods
                # such as object detection to identify image regions
                # For now, we'll use the whole page as an image
                result.append({"page": page_num, "image": page_image})

            return result
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
            return []

    def _generate_description(self, image: Image.Image, llm: Any) -> str:
        """Generate a description for an image using a vision-capable LLM."""
        try:
            # Encode image to base64 for API
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

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
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating image description: {e}")
            return "Image description unavailable"
