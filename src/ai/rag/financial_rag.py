"""Financial RAG Service for enhanced AI context."""
import hashlib
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.core.config import settings

logger = structlog.get_logger()

# Collection names
COLLECTION_MARKET_KNOWLEDGE = "market_knowledge"
COLLECTION_USER_CONTEXT = "user_context"
COLLECTION_FINANCIAL_EDUCATION = "financial_education"


class EmbeddingService:
    """Embedding service using sentence-transformers (local, no API required)."""

    def __init__(self):
        self._model = None
        self._model_name = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions

    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                logger.info("Loaded embedding model", model=self._model_name)
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
                raise
        return self._model

    @property
    def vector_size(self) -> int:
        """Get the vector size for the current model."""
        return 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text (synchronous)."""
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def get_embedding_async(self, text: str) -> list[float]:
        """Get embedding vector for text (async wrapper)."""
        # sentence-transformers is synchronous, but we can run in executor if needed
        # For now, just call synchronously as it's fast enough
        return self.get_embedding(text)


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


class FinancialRAGService:
    """RAG service for financial knowledge retrieval."""

    def __init__(self):
        self._embedding_service = get_embedding_service()
        self.VECTOR_SIZE = self._embedding_service.vector_size

        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=getattr(settings, "QDRANT_API_KEY", None) or None,
        )
        self._ensure_collections()

    def _ensure_collections(self):
        """Ensure all required collections exist."""
        collections = [
            COLLECTION_MARKET_KNOWLEDGE,
            COLLECTION_USER_CONTEXT,
            COLLECTION_FINANCIAL_EDUCATION,
        ]

        existing = [c.name for c in self.client.get_collections().collections]

        for collection in collections:
            if collection not in existing:
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection}")

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text using local model."""
        return await self._embedding_service.get_embedding_async(text)

    async def add_market_knowledge(
        self,
        content: str,
        source: str,
        category: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add market knowledge to RAG."""
        embedding = await self.get_embedding(content)

        point_id = str(uuid4())
        content_hash = hashlib.md5(content.encode()).hexdigest()

        payload = {
            "content": content,
            "source": source,
            "category": category,
            "content_hash": content_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        self.client.upsert(
            collection_name=COLLECTION_MARKET_KNOWLEDGE,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        logger.info(
            "Added market knowledge",
            point_id=point_id,
            source=source,
            category=category,
        )

        return point_id

    async def add_financial_education(
        self,
        topic: str,
        content: str,
        difficulty: str = "intermediate",
        tags: Optional[list[str]] = None,
    ) -> str:
        """Add financial education content."""
        embedding = await self.get_embedding(f"{topic}\n\n{content}")

        point_id = str(uuid4())

        payload = {
            "topic": topic,
            "content": content,
            "difficulty": difficulty,
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self.client.upsert(
            collection_name=COLLECTION_FINANCIAL_EDUCATION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        return point_id

    async def add_user_context(
        self,
        user_id: int,
        context_type: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add user-specific context for personalized responses."""
        embedding = await self.get_embedding(content)

        point_id = str(uuid4())

        payload = {
            "user_id": user_id,
            "context_type": context_type,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        self.client.upsert(
            collection_name=COLLECTION_USER_CONTEXT,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        return point_id

    async def search_market_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Search market knowledge base."""
        embedding = await self.get_embedding(query)

        filter_conditions = None
        if category:
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category),
                    )
                ]
            )

        results = self.client.search(
            collection_name=COLLECTION_MARKET_KNOWLEDGE,
            query_vector=embedding,
            query_filter=filter_conditions,
            limit=limit,
        )

        return [
            {
                "content": hit.payload.get("content"),
                "source": hit.payload.get("source"),
                "category": hit.payload.get("category"),
                "score": hit.score,
            }
            for hit in results
        ]

    async def search_education(
        self,
        query: str,
        difficulty: Optional[str] = None,
        limit: int = 3,
    ) -> list[dict]:
        """Search financial education content."""
        embedding = await self.get_embedding(query)

        filter_conditions = None
        if difficulty:
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="difficulty",
                        match=MatchValue(value=difficulty),
                    )
                ]
            )

        results = self.client.search(
            collection_name=COLLECTION_FINANCIAL_EDUCATION,
            query_vector=embedding,
            query_filter=filter_conditions,
            limit=limit,
        )

        return [
            {
                "topic": hit.payload.get("topic"),
                "content": hit.payload.get("content"),
                "difficulty": hit.payload.get("difficulty"),
                "score": hit.score,
            }
            for hit in results
        ]

    async def search_user_context(
        self,
        user_id: int,
        query: str,
        context_type: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Search user-specific context."""
        embedding = await self.get_embedding(query)

        must_conditions = [
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            )
        ]

        if context_type:
            must_conditions.append(
                FieldCondition(
                    key="context_type",
                    match=MatchValue(value=context_type),
                )
            )

        results = self.client.search(
            collection_name=COLLECTION_USER_CONTEXT,
            query_vector=embedding,
            query_filter=Filter(must=must_conditions),
            limit=limit,
        )

        return [
            {
                "content": hit.payload.get("content"),
                "context_type": hit.payload.get("context_type"),
                "score": hit.score,
            }
            for hit in results
        ]

    async def get_enriched_context(
        self,
        query: str,
        user_id: Optional[int] = None,
        user_profile: Optional[dict] = None,
    ) -> str:
        """Get enriched context for AI response."""
        contexts = []

        # Search market knowledge
        market_results = await self.search_market_knowledge(query, limit=3)
        if market_results:
            market_context = "\n\n".join([r["content"] for r in market_results])
            contexts.append(f"## Informacoes de Mercado\n{market_context}")

        # Search education content
        education_results = await self.search_education(query, limit=2)
        if education_results:
            edu_context = "\n\n".join(
                [f"**{r['topic']}**: {r['content']}" for r in education_results]
            )
            contexts.append(f"## Contexto Educacional\n{edu_context}")

        # Search user context if available
        if user_id:
            user_results = await self.search_user_context(user_id, query, limit=2)
            if user_results:
                user_context = "\n\n".join([r["content"] for r in user_results])
                contexts.append(f"## Contexto do Usuario\n{user_context}")

        # Add user profile context
        if user_profile:
            profile_context = f"""
## Perfil do Investidor
- Perfil de Risco: {user_profile.get('risk_profile', 'Nao definido')}
- Horizonte de Investimento: {user_profile.get('investment_horizon', 'Nao definido')}
- Experiencia: {user_profile.get('experience_level', 'Nao definido')}
"""
            contexts.append(profile_context)

        return "\n\n---\n\n".join(contexts) if contexts else ""

    def delete_user_context(self, user_id: int):
        """Delete all context for a user (LGPD compliance)."""
        self.client.delete(
            collection_name=COLLECTION_USER_CONTEXT,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            ),
        )
        logger.info("Deleted user context", user_id=user_id)


# Global instance
rag_service = FinancialRAGService()
