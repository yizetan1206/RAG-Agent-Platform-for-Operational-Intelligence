from pathlib import Path
from typing import List
from PyPDF2 import PdfReader

class DocumentLoader:
    """Load PDFs and text files from a directory."""

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder {folder_path} does not exist")

    def load(self) -> List[dict]:
        documents = []
        for file_path in self.folder_path.glob("*"):
            if file_path.suffix.lower() == ".pdf":
                text = self._load_pdf(file_path)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                text = self._load_text(file_path)
            else:
                continue
            if text.strip():
                documents.append({"source": str(file_path), "content": text})
        return documents

    def _load_pdf(self, file_path: Path) -> str:
        reader = PdfReader(file_path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)

    def _load_text(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8")
