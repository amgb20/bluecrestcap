"""Helpers for extracting text from PDFs with Mistral OCR."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests


def extract_pdf_text(
    file_path: str,
    api_key: Optional[str],
    model: str,
    endpoint: str,
    use_fallback: bool = False,
    timeout: int = 60,
) -> str:
    """Return text extracted from a PDF using Mistral's OCR service."""
    if use_fallback or not api_key:
        return _fallback_pdf_text(file_path)

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model}

    with open(file_path, "rb") as file_obj:
        files = {"file": (Path(file_path).name, file_obj, "application/pdf")}
        response = requests.post(
            endpoint,
            headers=headers,
            data=data,
            files=files,
            timeout=timeout,
        )

    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict):
        if "text" in payload and isinstance(payload["text"], str):
            return payload["text"]
        if "pages" in payload and isinstance(payload["pages"], list):
            return "\n\n".join(
                page.get("text", "") for page in payload["pages"] if isinstance(page, dict)
            )
        if "result" in payload and isinstance(payload["result"], list):
            return "\n\n".join(
                item.get("text", "") for item in payload["result"] if isinstance(item, dict)
            )

    raise ValueError("Unexpected OCR response format")


def _fallback_pdf_text(file_path: str) -> str:
    """Simple text fallback when OCR is not available."""
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return ""

    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
