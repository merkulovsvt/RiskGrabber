import asyncio
import logging
from typing import Callable, Optional, Sequence

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload

from .embeddings import get_embedder, INSTRUCT_BANK_REVIEW_PROMPT
from .sentiment import get_sentiment_classifier
from .vector_store import upsert_review_vectors_async
from ..backend import models

logger = logging.getLogger(__name__)


def embed_new_reviews(
        db: Session,
        batch_size: int = 10,
        progress_callback: Optional[Callable[[str, str, dict], None]] = None,
) -> int:
    async def _run() -> int:
        from ..backend.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            return await embed_new_reviews_async(session, batch_size, progress_callback)

    return asyncio.run(_run())


async def embed_new_reviews_async(
        db: AsyncSession,
        batch_size: int = 10,
        progress_callback: Optional[Callable[[str, str, dict], None]] = None,
) -> int:
    def report(stage: str, message: str, detail: Optional[dict] = None) -> None:
        if progress_callback:
            progress_callback(stage, message, detail or {})

    embedder = get_embedder()
    sentiment_clf = get_sentiment_classifier()
    updated_total = 0
    batch_num = 0

    while True:
        review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
        result = await db.execute(
            select(models.Review)
            .where(models.Review.vector_in_qdrant.is_(False))
            .order_by(review_date.asc())
            .limit(batch_size)
            .options(selectinload(models.Review.bank))
        )
        reviews: Sequence[models.Review] = result.scalars().all()
        if not reviews:
            break

        batch_num += 1
        report("embeddings",
               f"Эмбеддинги: батч {batch_num}, отзывов {len(reviews)} (всего обработано {updated_total})...",
               {"batch": batch_num, "batch_size": len(reviews), "total_so_far": updated_total})

        texts = []
        for r in reviews:
            texts.append(f"Отзыв о банке {r.bank.name.strip()}. Отзыв клиента (оценка 1–5): {r.rating}. Тема {r.title.strip()}. {r.text.strip()}")

        vectors = await asyncio.to_thread(
            lambda: embedder.embed(texts, prompt=INSTRUCT_BANK_REVIEW_PROMPT)
        )
        for review, vec in zip(reviews, vectors):
            label, score = await asyncio.to_thread(sentiment_clf.classify_vector, vec)
            review.sentiment = label
            review.sentiment_score = score

        n = await upsert_review_vectors_async(db, list(reviews), vectors)
        if n == 0:
            logger.error("Ошибка записи эмбеддингов в Qdrant для батча %s, прерываем чтобы не зациклиться", batch_num)
            break
        updated_total += n
        report("embeddings", f"Эмбеддинги: обработано {updated_total} отзывов",
               {"total": updated_total, "batch": batch_num})

    if progress_callback:
        report("embeddings", f"Эмбеддинги готовы. Всего обработано: {updated_total}",
               {"done": True, "total": updated_total})
    return updated_total
