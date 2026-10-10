"""Dense vector retriever using Qdrant vector database and Google Gemini embeddings."""

import os
import time
from typing import List, Optional, Callable
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.logging import logger
from app.retriever.base import BaseRetriever


class QdrantRetriever(BaseRetriever):
    """Dense retriever powered by Qdrant Vector Database and Gemini embeddings."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is required for QdrantRetriever.")

        self.embedding = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=key,
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.client: QdrantClient = self._init_client()
        self.vectorstore: Optional[QdrantVectorStore] = None
        self._init_vectorstore()

    def _init_client(self) -> QdrantClient:
        """Initialize Qdrant client for either remote server or persistent local disk storage."""
        if settings.QDRANT_URL:
            logger.info(f"Connecting to remote Qdrant cluster at {settings.QDRANT_URL}")
            return QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            logger.info(f"Initializing local persistent Qdrant storage at {settings.QDRANT_PATH}")
            for attempt in range(3):
                try:
                    return QdrantClient(path=settings.QDRANT_PATH)
                except Exception as e:
                    if "already accessed by another instance" in str(e) and attempt < 2:
                        time.sleep(1.0)
                        continue
                    raise e

    def _init_vectorstore(self):
        """Initialize QdrantVectorStore if collection exists."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if exists:
                self.vectorstore = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embedding,
                )
                logger.info(f"Attached to existing Qdrant collection: '{self.collection_name}'")
        except Exception as e:
            logger.warning(f"Could not check Qdrant collections: {e}")

    def create_or_update(
        self,
        documents: List[Document],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """Create or update Qdrant collection with batching for API rate safety."""
        if not documents:
            return

        batch_size = 10
        total_docs = len(documents)
        total_batches = ((total_docs - 1) // batch_size) + 1

        if progress_callback:
            progress_callback(f"Embedding batch 1/{total_batches} into Qdrant...", 0.0)

        first_batch = documents[:batch_size]

        # Check if collection exists; create if needed
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists or self.vectorstore is None:
            logger.info(f"Creating new Qdrant collection '{self.collection_name}'...")
            self.vectorstore = QdrantVectorStore.from_documents(
                documents=first_batch,
                embedding=self.embedding,
                client=self.client,
                collection_name=self.collection_name,
            )
        else:
            self.vectorstore.add_documents(first_batch)

        for i in range(batch_size, total_docs, batch_size):
            batch_num = (i // batch_size) + 1
            if progress_callback:
                progress_callback(
                    f"Embedding batch {batch_num}/{total_batches} into Qdrant...",
                    batch_num / total_batches,
                )
            time.sleep(3)  # Rate limit avoidance for Gemini free tier
            batch = documents[i : i + batch_size]
            self.vectorstore.add_documents(batch)

        if progress_callback:
            progress_callback("Qdrant indexing complete.", 1.0)
        logger.info(f"Successfully indexed {total_docs} document chunks in Qdrant collection '{self.collection_name}'.")

    def retrieve(self, query: str, top_k: int = 15) -> List[Document]:
        """Retrieve top_k documents by cosine vector similarity from Qdrant."""
        if self.vectorstore is None:
            # Try re-attaching if collection exists
            self._init_vectorstore()
            if self.vectorstore is None:
                return []
        try:
            return self.vectorstore.similarity_search(query, k=top_k)
        except Exception as e:
            logger.error(f"Error in Qdrant retrieval: {e}")
            return []

    def get_document_count(self) -> int:
        """Return the count of vectors currently stored in Qdrant collection."""
        try:
            if self.client:
                info = self.client.get_collection(self.collection_name)
                return info.points_count or 0
        except Exception:
            pass
        return 0
