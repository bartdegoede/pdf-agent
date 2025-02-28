import base64
import io
from typing import Any, Dict, List, Optional

import camelot
from langchain_openai import ChatOpenAI
from pdf2image import convert_from_path


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
        try:
            # Convert pages parameter to format Camelot expects
            if pages == "all":
                pages = "1-end"

            # Extract tables
            tables_data = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")

            # Process extracted tables
            result = []
            for i, table in enumerate(tables_data):
                # Convert to pandas DataFrame
                df = table.df

                # Get page number
                page_num = table.page

                # Convert to markdown
                markdown = df.to_markdown(index=False)

                result.append(
                    {
                        "page": page_num,
                        "markdown": markdown,
                        "data": df.to_dict(orient="records"),
                    }
                )

            return result
        except Exception as e:
            print(f"Error extracting tables with Camelot: {e}")
            return []

    def _extract_with_llm(
        self, pdf_path: str, llm: Any, pages: str
    ) -> List[Dict[str, Any]]:
        """Extract tables using a vision-capable LLM."""
        try:
            # Convert PDF to images
            if pages == "all":
                images = convert_from_path(pdf_path)
                page_indices = list(range(len(images)))
            else:
                # Parse pages string into list of page indices (0-based)
                page_nums = []
                for part in pages.split(","):
                    if "-" in part:
                        start, end = map(int, part.split("-"))
                        page_nums.extend(range(start, end + 1))
                    else:
                        page_nums.append(int(part))
                page_indices = [num - 1 for num in page_nums]  # Convert to 0-based
                images = convert_from_path(pdf_path)
                images = [images[i] for i in page_indices if i < len(images)]

            result = []

            for i, img in enumerate(images):
                page_num = page_indices[i] + 1  # Convert back to 1-based

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
                response = llm.invoke(messages)
                markdown_tables = response.content

                # If tables were found
                if "No tables found" not in markdown_tables:
                    result.append(
                        {
                            "page": page_num,
                            "markdown": markdown_tables,
                            "data": None,  # We don't have structured data from LLM extraction
                        }
                    )

            return result
        except Exception as e:
            print(f"Error extracting tables with LLM: {e}")
            return []
