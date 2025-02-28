from unittest.mock import MagicMock, patch

from pdf_mind.tools.pdf_reader import PDFReaderTool


class TestPDFReaderTool:
    """Tests for the PDFReaderTool class."""

    def test_init(self):
        """Test that the tool can be initialized."""
        tool = PDFReaderTool()
        assert isinstance(tool, PDFReaderTool)

    @patch("pdf_extraction_agent.tools.pdf_reader.pypdf.PdfReader")
    def test_extract_with_pypdf(self, mock_pdf_reader):
        """Test extracting text with PyPDF."""
        # Set up mock
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is test text."
        mock_pdf_reader.return_value.pages = [mock_page]

        # Create tool and extract text
        tool = PDFReaderTool()
        text = tool._extract_with_pypdf("test.pdf")

        # Check results
        assert "This is test text." in text
        mock_pdf_reader.assert_called_once()
        mock_page.extract_text.assert_called_once()

    @patch("pdf_extraction_agent.tools.pdf_reader.pypdf.PdfReader")
    def test_extract_text_pypdf_success(self, mock_pdf_reader):
        """Test extract_text when PyPDF succeeds."""
        # Set up mock
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is test text."
        mock_pdf_reader.return_value.pages = [mock_page]

        # Create tool and extract text
        tool = PDFReaderTool()
        text = tool.extract_text("test.pdf", fallback_to_llm_ocr=False)

        # Check results
        assert "This is test text." in text

    @patch("pdf_extraction_agent.tools.pdf_reader.PDFReaderTool._extract_with_pypdf")
    @patch("pdf_extraction_agent.tools.pdf_reader.PDFReaderTool._extract_with_llm_ocr")
    def test_extract_text_fallback_to_llm(self, mock_llm_ocr, mock_pypdf):
        """Test that extract_text falls back to LLM OCR when PyPDF fails."""
        # Set up mocks
        mock_pypdf.return_value = ""  # PyPDF fails
        mock_llm_ocr.return_value = "Text extracted with LLM OCR."

        # Create tool and extract text
        tool = PDFReaderTool()
        text = tool.extract_text("test.pdf", llm=MagicMock())

        # Check results
        assert text == "Text extracted with LLM OCR."
        mock_pypdf.assert_called_once()
        mock_llm_ocr.assert_called_once()

    @patch("pdf_extraction_agent.tools.pdf_reader.PDFReaderTool._extract_with_pypdf")
    @patch("pdf_extraction_agent.tools.pdf_reader.PDFReaderTool._extract_with_llm_ocr")
    def test_no_fallback_when_disabled(self, mock_llm_ocr, mock_pypdf):
        """Test that OCR fallback is not used when disabled."""
        # Set up mocks
        mock_pypdf.return_value = ""  # PyPDF fails

        # Create tool and extract text with fallback disabled
        tool = PDFReaderTool()
        text = tool.extract_text("test.pdf", fallback_to_llm_ocr=False)

        # Check results
        assert text == ""
        mock_pypdf.assert_called_once()
        mock_llm_ocr.assert_not_called()
