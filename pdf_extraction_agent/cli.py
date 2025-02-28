import sys
from typing import Optional

import click

from pdf_extraction_agent import PDFExtractionAgent
from pdf_extraction_agent.config import PDFExtractionConfig


@click.group()
def cli():
    """PDF Extraction Agent CLI."""
    pass


@cli.command()
@click.argument(
    "pdf_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Path to save the extracted content to. If not provided, prints to stdout.",
)
@click.option(
    "--model", "-m", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)"
)
@click.option("--no-tables", is_flag=True, help="Skip table extraction")
@click.option("--no-images", is_flag=True, help="Skip image extraction")
@click.option("--no-llm-ocr", is_flag=True, help="Disable LLM-based OCR fallback")
@click.option("--save-images", is_flag=True, help="Save extracted images to disk")
@click.option(
    "--image-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory to save extracted images to",
)
def extract(
    pdf_path: str,
    output: Optional[str],
    model: str,
    no_tables: bool,
    no_images: bool,
    no_llm_ocr: bool,
    save_images: bool,
    image_dir: Optional[str],
) -> int:
    """Extract structured content from PDFs using AI.

    PDF_PATH is the path to the PDF file to process.
    """
    # Create config
    config = PDFExtractionConfig(
        openai_model=model,
        use_llm_ocr=not no_llm_ocr,
        extract_tables=not no_tables,
        extract_images=not no_images,
        save_images=save_images,
        output_dir=image_dir,
    )

    # Create agent
    try:
        agent = PDFExtractionAgent(
            openai_api_key=config.get_openai_api_key(),
            openai_model=config.openai_model,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return 1

    # Process PDF
    try:
        click.echo("Processing PDF, this may take a while...")
        result = agent.process(pdf_path)

        # Output result
        if output:
            with open(output, "w") as f:
                f.write(result)
            click.echo(f"Extracted content saved to {output}")
        else:
            click.echo(result)

        return 0
    except Exception as e:
        click.echo(f"Error processing PDF: {e}", err=True)
        return 1


@cli.command()
@click.option("--version", is_flag=True, help="Show version information")
def info(version: bool) -> int:
    """Show information about the PDF Extraction Agent."""
    if version:
        from importlib.metadata import version as get_version

        try:
            version_info = get_version("pdf-extraction-agent")
            click.echo(f"PDF Extraction Agent version: {version_info}")
        except Exception:
            click.echo("PDF Extraction Agent (version unknown)")
    else:
        click.echo("PDF Extraction Agent")
        click.echo(
            "A tool for extracting structured content from PDFs using LangGraph and OpenAI."
        )
        click.echo("\nUse 'pdf-extract extract --help' for more information.")

    return 0


def main() -> int:
    """Main entry point for the CLI."""
    return cli()


if __name__ == "__main__":
    sys.exit(main())
