import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openai import OpenAI

from pdf_extraction_agent.tools.image_extractor import ImageExtractorTool
from pdf_extraction_agent.tools.pdf_reader import PDFReaderTool
from pdf_extraction_agent.tools.table_extractor import TableExtractorTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pdf_extraction_agent")


# Define the state type using TypedDict
class PDFExtractionState(TypedDict, total=False):
    """State type for the PDF extraction workflow."""

    pdf_path: str
    text: Optional[str]
    tables: Optional[List[Dict[str, Any]]]
    images: Optional[List[Dict[str, Any]]]
    final_content: Optional[str]


class PDFExtractionAgent:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o",
    ):
        """Initialize the PDF Extraction Agent.

        Args:
            openai_api_key: OpenAI API key. If None, it will be read from the
                OPENAI_API_KEY env var.
            openai_model: OpenAI model to use.
        """
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model

        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model=openai_model,
            api_key=openai_api_key,
            temperature=0,
        )

        # Initialize direct OpenAI client for token counting
        self.openai_client = OpenAI(api_key=openai_api_key)

        self.tools = {
            "pdf_reader": PDFReaderTool(),
            "table_extractor": TableExtractorTool(),
            "image_extractor": ImageExtractorTool(),
        }
        self.workflow = self._create_workflow()
        self.last_result = None  # Store the last workflow result

        # Token usage tracking
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for PDF extraction."""
        workflow = StateGraph(PDFExtractionState)

        # Define the nodes
        workflow.add_node("extract_text", self._extract_text)
        workflow.add_node("extract_tables", self._extract_tables)
        workflow.add_node("extract_images", self._extract_images)
        workflow.add_node("combine_results", self._combine_results)

        # Define the edges
        workflow.add_edge("extract_text", "extract_tables")
        workflow.add_edge("extract_tables", "extract_images")
        workflow.add_edge("extract_images", "combine_results")
        workflow.add_edge("combine_results", END)

        # Set the entry point
        workflow.set_entry_point("extract_text")

        return workflow.compile()

    async def _extract_text(self, state: PDFExtractionState) -> PDFExtractionState:
        """Extract text from the PDF."""
        pdf_path = state["pdf_path"]
        logger.info("Starting text extraction from PDF: %s", pdf_path)
        start_time = time.time()
        try:
            text = self.tools["pdf_reader"].extract_text(pdf_path)
            elapsed = time.time() - start_time
            logger.info("Text extraction completed in %.2f seconds", elapsed)
            return {"pdf_path": pdf_path, "text": text}
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}", exc_info=True)
            raise

    async def _extract_tables(self, state: PDFExtractionState) -> PDFExtractionState:
        """Extract tables from the PDF."""
        pdf_path = state["pdf_path"]
        logger.info("Starting table extraction from PDF: %s", pdf_path)
        start_time = time.time()
        try:
            tables = self.tools["table_extractor"].extract_tables(pdf_path)
            elapsed = time.time() - start_time
            logger.info(
                "Table extraction completed in %.2f seconds, found %d tables",
                elapsed,
                len(tables),
            )
            return {**state, "tables": tables}
        except Exception as e:
            logger.error("Table extraction failed: %s", str(e), exc_info=True)
            raise

    async def _extract_images(self, state: PDFExtractionState) -> PDFExtractionState:
        """Extract images with descriptions from the PDF."""
        pdf_path = state["pdf_path"]
        logger.info("Starting image extraction from PDF: %s", pdf_path)
        start_time = time.time()
        try:
            images = self.tools["image_extractor"].extract_images(pdf_path, self.llm)
            elapsed = time.time() - start_time
            logger.info(
                "Image extraction completed in %.2f seconds, found %d images",
                elapsed,
                len(images),
            )
            return {**state, "images": images}
        except Exception as e:
            logger.error("Image extraction failed: %s", str(e), exc_info=True)
            raise

    async def _combine_results(self, state: PDFExtractionState) -> PDFExtractionState:
        """Combine all extracted elements into a structured result."""
        logger.info("Starting combination of extracted elements")
        start_time = time.time()
        try:
            prompt = self._create_combination_prompt(state)
            logger.info("Created combination prompt (length: %d chars)", len(prompt))

            system_content = (
                "You are a PDF content organizer. Your task is to combine text, "
                "tables, and images into a well-structured document."
            )

            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=prompt),
            ]

            # Create messages in OpenAI format for token counting
            openai_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            # Get token count for this request
            tokens_response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=openai_messages,
                max_tokens=1,  # Minimize tokens for just counting
                temperature=0,
            )

            # Update token counts
            self.token_usage["prompt_tokens"] += tokens_response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += (
                tokens_response.usage.completion_tokens
            )
            self.token_usage["total_tokens"] += tokens_response.usage.total_tokens
            self.token_usage["api_calls"] += 1

            logger.info(
                "Token usage for combination - Prompt: %d, Completion: %d, Total: %d",
                tokens_response.usage.prompt_tokens,
                tokens_response.usage.completion_tokens,
                tokens_response.usage.total_tokens,
            )

            # Now make the actual call for the real content
            logger.info("Calling LLM to combine elements")
            response = await self.llm.ainvoke(messages)

            # Add completion tokens for the actual response (estimate)
            self.token_usage["api_calls"] += 1

            elapsed = time.time() - start_time
            logger.info("Results combination completed in %.2f seconds", elapsed)
            return {**state, "final_content": response.content}
        except Exception as e:
            logger.error("Results combination failed: %s", str(e), exc_info=True)
            raise

    def _create_combination_prompt(self, state: PDFExtractionState) -> str:
        """Create a prompt for combining the extracted elements."""
        prompt = f"""I have extracted the following elements from a PDF:

TEXT:
{state.get('text', 'No text extracted')}

TABLES:
{self._format_tables(state.get('tables', []))}

IMAGES:
{self._format_images(state.get('images', []))}

Please combine these elements into a well-structured document, maintaining the logical flow.
Place tables and images near related text. Use markdown formatting.
"""
        return prompt

    def _format_tables(self, tables: List[Dict[str, Any]]) -> str:
        """Format extracted tables for the prompt."""
        if not tables:
            return "No tables extracted"

        result = ""
        for i, table in enumerate(tables):
            result += f"Table {i+1}:\n{table['markdown']}\n\n"
        return result

    def _format_images(self, images: List[Dict[str, Any]]) -> str:
        """Format extracted images for the prompt."""
        if not images:
            return "No images extracted"

        result = ""
        for i, image in enumerate(images):
            result += f"Image {i+1}: {image['description']}\n\n"
        return result

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics from the most recent extraction.

        Returns:
            Dictionary containing extraction statistics or None if no extraction has been performed.
        """
        if self.last_result is None:
            logger.warning("No extraction has been performed yet")
            return None

        # Calculate stats from the last result
        stats = {
            "table_count": len(self.last_result.get("tables", [])),
            "image_count": len(self.last_result.get("images", [])),
            "content_length": len(self.last_result.get("final_content", "")),
            "text_length": len(self.last_result.get("text", "")),
            "has_tables": len(self.last_result.get("tables", [])) > 0,
            "has_images": len(self.last_result.get("images", [])) > 0,
        }

        # Add table details if present
        if stats["has_tables"]:
            table_pages = [
                table.get("page", "unknown")
                for table in self.last_result.get("tables", [])
            ]
            stats["table_pages"] = table_pages

        # Add image details if present
        if stats["has_images"]:
            image_pages = [
                image.get("page", "unknown")
                for image in self.last_result.get("images", [])
            ]
            stats["image_pages"] = image_pages

        # Add token usage information
        stats["token_usage"] = {
            "prompt_tokens": self.token_usage["prompt_tokens"],
            "completion_tokens": self.token_usage["completion_tokens"],
            "total_tokens": self.token_usage["total_tokens"],
            "api_calls": self.token_usage["api_calls"],
        }

        return stats

    async def aprocess(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF and extract structured content asynchronously.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict containing the structured content and extraction stats.
        """
        # Reset token usage for this run
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }

        logger.info("Starting asynchronous processing of PDF: %s", pdf_path)
        start_time = time.time()
        try:
            result = await self.workflow.ainvoke({"pdf_path": pdf_path})
            elapsed = time.time() - start_time

            # Store the complete workflow result
            self.last_result = result

            # Create extraction stats
            extraction_stats = {
                "total_time": elapsed,
                "table_count": len(result.get("tables", [])),
                "image_count": len(result.get("images", [])),
                "content_length": len(result.get("final_content", "")),
                "text_length": len(result.get("text", "")),
                "token_usage": {
                    "prompt_tokens": self.token_usage["prompt_tokens"],
                    "completion_tokens": self.token_usage["completion_tokens"],
                    "total_tokens": self.token_usage["total_tokens"],
                    "api_calls": self.token_usage["api_calls"],
                },
            }

            logger.info(
                f"PDF processing completed in {elapsed:.2f} seconds. "
                f"Found {extraction_stats['table_count']} tables and {extraction_stats['image_count']} images."
            )
            logger.info(
                f"Token usage - Prompt: {self.token_usage['prompt_tokens']}, "
                f"Completion: {self.token_usage['completion_tokens']}, "
                f"Total: {self.token_usage['total_tokens']} "
                f"across {self.token_usage['api_calls']} API calls"
            )

            return {"content": result["final_content"], "stats": extraction_stats}
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"PDF processing failed after {elapsed:.2f} seconds: {str(e)}",
                exc_info=True,
            )
            raise

    def process(self, pdf_path: str) -> Dict[str, Any]:
        """Process a PDF and extract structured content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict containing the structured content and extraction stats.
        """
        import asyncio

        logger.info("Starting synchronous processing of PDF: %s", pdf_path)
        start_time = time.time()

        try:
            # Create a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    logger.info("Existing event loop is closed, creating new one")
                    raise RuntimeError("Event loop is closed")
                logger.info("Using existing event loop")
            except (RuntimeError, ValueError):
                logger.info("Creating new event loop")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the async process function
            logger.info("Running async workflow in event loop")
            result = loop.run_until_complete(self.aprocess(pdf_path))

            elapsed = time.time() - start_time
            logger.info(
                f"Synchronous PDF processing completed in {elapsed:.2f} seconds"
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Synchronous PDF processing failed after {elapsed:.2f} seconds: {str(e)}",
                exc_info=True,
            )
            raise
