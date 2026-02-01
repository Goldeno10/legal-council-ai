from docling.document_converter import DocumentConverter


def parse_legal_document(file_path: str) -> str:
    """
    Converts complex legal PDFs into structured Markdown to preserve 
    clause hierarchy and table data.

    Example usage:
    md_content = parse_legal_document("employment_contract.pdf")

    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
    # Export to markdown for better LLM reasoning
    return result.document.export_to_markdown()

