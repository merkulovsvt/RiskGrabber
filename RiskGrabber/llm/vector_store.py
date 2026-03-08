import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func

from ..backend.config import get_settings
from ..backend import models
from .embeddings import get_embedder, INSTRUCT_BANK_REVIEW_PROMPT
from .sentiment import get_sentiment_classifier

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(host=s.qdrant_host, port=s.qdrant_port)


def ensure_collection(vector_size: int) -> None:
    try:
        client = get_qdrant_client()
    except Exception as e:
        logger.warning("Не удалось подключиться к Qdrant: %s. Продолжаем без Qdrant.", e)
        return

    try:
        if client.collection_exists(get_settings().qdrant_collection):
            return
    except Exception as e:
        logger.warning("Проверка коллекции Qdrant: %s. Пропускаем создание.", e)
        return

    logger.info("Создание коллекции Qdrant '%s' (без пересоздания).", get_settings().qdrant_collection)
    try:
        client.create_collection(
            collection_name=get_settings().qdrant_collection,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
    except Exception as e2:
        logger.warning("Не удалось создать коллекцию Qdrant: %s. Продолжаем без Qdrant.", e2)


def upsert_review_vectors(
    db: Session,
    reviews: List[models.Review],
    vectors: List[List[float]],
) -> int:
    if not reviews or not vectors or len(reviews) != len(vectors):
        return 0
    try:
        client = get_qdrant_client()
    except Exception as e:
        logger.warning("Qdrant недоступен: %s", e)
        return 0
    dim = len(vectors[0])
    try:
        ensure_collection(dim)
    except Exception as e:
        logger.warning("Не удалось создать/проверить коллекцию Qdrant: %s", e)
        return 0
    points: List[qmodels.PointStruct] = []
    for r, vec in zip(reviews, vectors):
        review_id = int(r.id)
        payload = {
            "review_id": review_id,
            "bank_id": r.bank_id,
            "rating": r.rating,
            "sentiment": r.sentiment,
            "sentiment_score": r.sentiment_score,
            "published_at": r.published_at.isoformat() if r.published_at else None,
            "scraped_at": r.scraped_at.isoformat() if r.scraped_at else None,
            "title": r.title,
            "text_preview": r.text
        }
        points.append(qmodels.PointStruct(id=review_id, vector=vec, payload=payload))
    try:
        client.upsert(collection_name=get_settings().qdrant_collection, points=points)
        for r in reviews:
            r.vector_in_qdrant = True
        db.commit()
        return len(points)
    except Exception as e:
        logger.warning("Ошибка при upsert в Qdrant: %s", e)
        db.rollback()
        return 0


def get_vectors_by_review_ids(review_ids: List[int]) -> Dict[int, List[float]]:
    if not review_ids:
        return {}
    try:
        client = get_qdrant_client()
    except Exception as e:
        logger.warning("Qdrant недоступен: %s", e)
        return {}
    try:
        points = client.retrieve(
            collection_name=get_settings().qdrant_collection,
            ids=review_ids,
            with_vectors=True,
            with_payload=False,
        )
        # Нормализуем id к int, чтобы lookup по review_id (int) всегда находил вектор
        result = {}
        for p in points:
            if p.vector is None:
                continue
            try:
                key = int(p.id)
            except (TypeError, ValueError):
                key = p.id
            result[key] = p.vector
        if not result and review_ids:
            logger.warning(
                "Qdrant retrieve вернул пустой результат для ids=%s (коллекция=%s). "
                "Проверьте, что точки записаны с теми же id.",
                review_ids[:20],
                get_settings().qdrant_collection,
            )
        return result
    except Exception as e:
        logger.warning("Не удалось загрузить векторы из Qdrant: %s", e)
        return {}


def get_vector_for_review(review_id: int) -> Optional[List[float]]:
    """Вектор одного отзыва из Qdrant."""
    d = get_vectors_by_review_ids([review_id])
    return d.get(review_id)


def ensure_risks_collection(vector_size: int) -> None:
    """Создаёт коллекцию для эмбеддингов рисков, если её ещё нет."""
    try:
        client = get_qdrant_client()
    except Exception as e:
        logger.warning("Qdrant недоступен для risks: %s", e)
        return
    coll = get_settings().qdrant_risks_collection
    try:
        if client.collection_exists(coll):
            return
    except Exception as e:
        logger.warning("Проверка коллекции рисков Qdrant: %s", e)
        return
    try:
        client.create_collection(
            collection_name=coll,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info("Создана коллекция Qdrant для рисков: %s", coll)
    except Exception as e:
        logger.warning("Не удалось создать коллекцию рисков: %s", e)


def upsert_risk_vector(risk_id: int, vector: List[float], title: str = "") -> bool:
    """Сохранить эмбеддинг риска в Qdrant (id точки = risk_id)."""
    if not vector:
        return False
    try:
        client = get_qdrant_client()
    except Exception as e:
        logger.warning("Qdrant недоступен: %s", e)
        return False
    ensure_risks_collection(len(vector))
    try:
        point = qmodels.PointStruct(
            id=risk_id,
            vector=vector,
            payload={"risk_id": risk_id, "title": (title or "")[:500]},
        )
        client.upsert(collection_name=get_settings().qdrant_risks_collection, points=[point])
        return True
    except Exception as e:
        logger.warning("Ошибка upsert вектора риска %s: %s", risk_id, e)
        return False


def search_risks_by_vector(
    vector: List[float],
    top_k: int,
) -> List[Tuple[int, float]]:
    """
    Поиск по коллекции рисков: возвращает (risk_id, score) для top_k ближайших.
    Если коллекция пуста или ошибка — возвращает [].
    """
    if not vector or top_k <= 0:
        return []
    try:
        client = get_qdrant_client()
    except Exception as e:
        logger.warning("Qdrant недоступен: %s", e)
        return []
    coll = get_settings().qdrant_risks_collection
    try:
        if not client.collection_exists(coll):
            return []
    except Exception:
        return []
    try:
        response = client.query_points(
            collection_name=coll,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        points = response
        if not isinstance(points, list):
            points = getattr(response, "points", None) or getattr(response, "result", []) or []
        out: List[Tuple[int, float]] = []
        for p in points:
            try:
                pid = getattr(p, "id", None)
                payload = getattr(p, "payload", None) or {}
                rid = int(payload.get("risk_id", pid)) if isinstance(payload, dict) else int(pid)
            except (TypeError, ValueError):
                rid = int(getattr(p, "id", 0))
            score = float(getattr(p, "score", 0) or 0)
            out.append((rid, score))
        return out
    except Exception as e:
        logger.warning("Ошибка поиска рисков по вектору: %s", e)
        return []


def sync_all_reviews_to_qdrant(db: Session, incremental: bool = True) -> int:
    batch_size = 8
    total = 0
    embedder = get_embedder()
    sentiment_clf = get_sentiment_classifier()
    while True:
        review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
        reviews = (
            db.query(models.Review)
            .filter(models.Review.vector_in_qdrant.is_(False))
            .order_by(review_date.asc())
            .limit(batch_size)
            .all()
        )
        if not reviews:
            break
        texts = []
        for r in reviews:
            texts.append(f"Отзыв о банке {r.bank.name.strip()}. Отзыв клиента (оценка 1–5): {r.rating}. Тема {r.title.strip()}. {r.text.strip()}")
        vectors = embedder.embed(texts, prompt=INSTRUCT_BANK_REVIEW_PROMPT)

        for r, vec in zip(reviews, vectors):
            label, score = sentiment_clf.classify_vector(vec)
            r.sentiment = label
            r.sentiment_score = score
        n = upsert_review_vectors(db, reviews, vectors)
        if n == 0:
            logger.warning("Qdrant upsert вернул 0, прерываем sync")
            break
        total += n
    if total:
        logger.info("Sync to Qdrant: %s reviews", total)
    return total


async def sync_all_reviews_to_qdrant_async(db: AsyncSession, incremental: bool = True) -> int:
    batch_size = 8
    total = 0
    embedder = get_embedder()
    sentiment_clf = get_sentiment_classifier()
    while True:
        review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
        result = await db.execute(
            select(models.Review)
            .where(models.Review.vector_in_qdrant.is_(False))
            .order_by(review_date.asc())
            .limit(batch_size)
            .options(selectinload(models.Review.bank))
        )
        reviews = result.scalars().all()
        if not reviews:
            break
        texts = []
        for r in reviews:
            texts.append(f"Отзыв о банке {r.bank.name.strip()}. Отзыв клиента (оценка 1–5): {r.rating}. Тема {r.title.strip()}. {r.text.strip()}")    
        vectors = await asyncio.to_thread(
            lambda: embedder.embed(texts, prompt=INSTRUCT_BANK_REVIEW_PROMPT)
        )
        
        for r, vec in zip(reviews, vectors):
            label, score = await asyncio.to_thread(sentiment_clf.classify_vector, vec)
            r.sentiment = label
            r.sentiment_score = score
        n = await upsert_review_vectors_async(db, reviews, vectors)
        if n == 0:
            logger.warning("Qdrant upsert вернул 0, прерываем sync")
            break
        total += n
    if total:
        logger.info("Sync to Qdrant: %s reviews", total)
    return total


async def get_vector_for_review_async(review_id: int) -> Optional[List[float]]:
    return await asyncio.to_thread(get_vector_for_review, review_id)


async def get_vectors_by_review_ids_async(review_ids: List[int]) -> Dict[int, List[float]]:
    return await asyncio.to_thread(get_vectors_by_review_ids, review_ids)


async def search_risks_by_vector_async(
    vector: List[float],
    top_k: int,
) -> List[Tuple[int, float]]:
    return await asyncio.to_thread(search_risks_by_vector, vector, top_k)


async def upsert_risk_vector_async(risk_id: int, vector: List[float], title: str = "") -> bool:
    return await asyncio.to_thread(upsert_risk_vector, risk_id, vector, title)


def _qdrant_upsert_review_points_only(points: List[qmodels.PointStruct], dim: int) -> int:
    try:
        client = get_qdrant_client()
    except Exception as e:
        logger.warning("Qdrant недоступен: %s", e)
        return 0
    ensure_collection(dim)
    try:
        client.upsert(collection_name=get_settings().qdrant_collection, points=points)
        return len(points)
    except Exception as e:
        logger.warning("Ошибка при upsert в Qdrant: %s", e)
        return 0


async def upsert_review_vectors_async(
    db: AsyncSession,
    reviews: List[models.Review],
    vectors: List[List[float]],
) -> int:
    if not reviews or not vectors or len(reviews) != len(vectors):
        return 0
    points: List[qmodels.PointStruct] = []
    for r, vec in zip(reviews, vectors):
        review_id = int(r.id)
        payload = {
            "review_id": review_id,
            "bank_id": r.bank_id,
            "rating": r.rating,
            "sentiment": r.sentiment,
            "sentiment_score": r.sentiment_score,
            "published_at": r.published_at.isoformat() if r.published_at else None,
            "scraped_at": r.scraped_at.isoformat() if r.scraped_at else None,
            "title": r.title,
            "text_preview": r.text
        }
        points.append(qmodels.PointStruct(id=review_id, vector=vec, payload=payload))
    dim = len(vectors[0])
    n = await asyncio.to_thread(_qdrant_upsert_review_points_only, points, dim)
    if n > 0:
        for r in reviews:
            r.vector_in_qdrant = True
        await db.commit()
    return n

