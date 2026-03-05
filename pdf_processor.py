import io
import re


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text from an uploaded PDF file.
    Uses PyPDF2 as primary extractor.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Extracted text as a single string
    """
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks for better retrieval.

    Why overlap? So context isn't lost at chunk boundaries.

    Args:
        text: Full document text
        chunk_size: Characters per chunk
        overlap: Characters of overlap between consecutive chunks

    Returns:
        List of text chunk strings
    """
    # Clean up excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to end chunk at a sentence boundary
        if end < len(text):
            # Look for '. ' or '\n' near the end
            boundary = text.rfind('. ', start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap  # overlap for context continuity

    return chunks
