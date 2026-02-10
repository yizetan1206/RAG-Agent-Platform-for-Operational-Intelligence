from pathlib import Path
from typing import List
import logging
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load PDFs and text files from a directory."""

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

        if not self.folder_path.exists():
            logger.error(
                "Document folder does not exist",
                extra={"path": str(self.folder_path)}
            )
            raise FileNotFoundError(f"Folder {folder_path} does not exist")

        logger.info(
            "DocumentLoader initialized",
            extra={"path": str(self.folder_path)}
        )

    def load(self) -> List[dict]:
        documents = []
        supported_ext = {".pdf", ".txt", ".md"}

        logger.info("Document loading started")

        for file_path in self.folder_path.glob("*"):
            if file_path.suffix.lower() not in supported_ext:
                logger.debug(
                    "Skipping unsupported file",
                    extra={"file": file_path.name}
                )
                continue

            try:
                if file_path.suffix.lower() == ".pdf":
                    text = self._load_pdf(file_path)
                else:
                    text = self._load_text(file_path)
            except Exception:
                logger.exception(
                    "Failed to load document",
                    extra={"file": file_path.name}
                )
                continue

            if not text.strip():
                logger.warning(
                    "Empty document skipped",
                    extra={"file": file_path.name}
                )
                continue

            documents.append(
                {
                    "source": str(file_path),
                    "content": text,
                }
            )

            logger.debug(
                "Document loaded",
                extra={"file": file_path.name}
            )

        logger.info(
            "Document loading completed",
            extra={"document_count": len(documents)}
        )

        return documents

    def _load_pdf(self, file_path: Path) -> str:
        reader = PdfReader(file_path)
        pages = [p.extract_text() or "" for p in reader.pages]

        logger.debug(
            "PDF parsed",
            extra={
                "file": file_path.name,
                "page_count": len(pages),
            }
        )

        return "\n".join(pages)

    def _load_text(self, file_path: Path) -> str:
        logger.debug(
            "Text file parsed",
            extra={"file": file_path.name}
        )

        return file_path.read_text(encoding="utf-8")
