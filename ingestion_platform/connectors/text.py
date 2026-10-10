"""Text connector for the ingestion pipeline."""

import os
from typing import List

from ingestion_platform.connectors.base import BaseConnector
from ingestion_platform.core.models import StandardDocument
from app.core.logging import logger

class TextConnector(BaseConnector):
    """Extracts text from TXT and MD markdown files."""

    def extract(self, source: str) -> List[StandardDocument]:
        """
        Extracts content from a TXT or MD file path.
        """
        if not os.path.exists(source):
            logger.error(f"Invalid text source: {source}")
            return []

        ext = source.split(".")[-1].lower()
        if ext not in ["txt", "md", "markdown"]:
            logger.error(f"Invalid text/markdown source: {source}")
            return []

        try:
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()

            return [
                StandardDocument(
                    content=content,
                    source_id=f"{source}_chunk_1",
                    connector_type="markdown" if ext in ["md", "markdown"] else "text",
                    metadata={
                        "original_source": source,
                        "source_type": f"file_{ext}"
                    }
                )
            ]
            
        except Exception as e:
            logger.error(f"Error extracting {source}: {e}")
            return []
