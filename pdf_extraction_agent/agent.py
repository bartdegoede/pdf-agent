from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from pdf_extraction_agent.tools.image_extractor import ImageExtractorTool
from pdf_extraction_agent.tools.pdf_reader import PDFReaderTool
from pdf_extraction_agent.tools.table_extractor import TableExtractorTool


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
            openai_api_key: OpenAI API key. If None, it will be read from the OPENAI_API_KEY env var.
            openai_model: OpenAI model to use.
        """
        self.llm = ChatOpenAI(
            model=openai_model,
            api_key=openai_api_key,
            temperature=0,
        )
        self.tools = {
            "pdf_reader": PDFReaderTool(),
            "table_extractor": TableExtractorTool(),
            "image_extractor": ImageExtractorTool(),
        }
        self.workflow = self._create_workflow()

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
        text = self.tools["pdf_reader"].extract_text(pdf_path)
        return {"pdf_path": pdf_path, "text": text}

    async def _extract_tables(self, state: PDFExtractionState) -> PDFExtractionState:
        """Extract tables from the PDF."""
        pdf_path = state["pdf_path"]
        tables = self.tools["table_extractor"].extract_tables(pdf_path)
        return {**state, "tables": tables}

    async def _extract_images(self, state: PDFExtractionState) -> PDFExtractionState:
        """Extract images with descriptions from the PDF."""
        pdf_path = state["pdf_path"]
        images = self.tools["image_extractor"].extract_images(pdf_path, self.llm)
        return {**state, "images": images}

    async def _combine_results(self, state: PDFExtractionState) -> PDFExtractionState:
        """Combine all extracted elements into a structured result."""
        prompt = self._create_combination_prompt(state)
        messages = [
            SystemMessage(
                content="You are a PDF content organizer. Your task is to combine text, "
                "tables, and images into a well-structured document."
            ),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)
        return {**state, "final_content": response.content}

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

    async def aprocess(self, pdf_path: str) -> str:
        """Process a PDF and extract structured content asynchronously.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Structured content extracted from the PDF.
        """
        result = await self.workflow.ainvoke({"pdf_path": pdf_path})
        return result["final_content"]

    def process(self, pdf_path: str) -> str:
        """Process a PDF and extract structured content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Structured content extracted from the PDF.
        """
        import asyncio

        # Create a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except (RuntimeError, ValueError):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async process function
        return loop.run_until_complete(self.aprocess(pdf_path))
