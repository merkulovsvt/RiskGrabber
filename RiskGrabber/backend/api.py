import asyncio
import datetime as dt
import json
import logging
import math
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional

import umap
import numpy as np
from fastapi import FastAPI, Depends, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import case, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload
from sklearn.decomposition import PCA

from .config import get_settings
from .database import AsyncSessionLocal, init_db, get_db, get_db_sync, SessionLocal
from .time_utils import moscow_now
from . import models, schemas
from .scraper import ingest_reviews, ingest_reviews_since
from .dataset_loader import load_russian_bank_reviews_into_db
from ..llm.embeddings import get_embedder
from ..llm.pipeline import embed_new_reviews, embed_new_reviews_async
from ..llm.risk_detection import generate_risk_for_review_async
from ..llm.vector_store import sync_all_reviews_to_qdrant, sync_all_reviews_to_qdrant_async, get_vectors_by_review_ids_async

logger = logging.getLogger(__name__)

# Агрегированный показатель рисков: байесовское среднее (1–5)
BAYESIAN_PRIOR_MEAN = 3.0
BAYESIAN_PRIOR_WEIGHT = 10


def _risk_score_bayesian(w: float, n_risk: int) -> float:
    """Байесовское среднее критичности: (w + m*C) / (n_risk + m), результат в [1, 5]."""
    if n_risk <= 0:
        return 0.0
    raw = (w + BAYESIAN_PRIOR_WEIGHT * BAYESIAN_PRIOR_MEAN) / (n_risk + BAYESIAN_PRIOR_WEIGHT)
    return max(1.0, min(5.0, round(raw, 4)))


def _unified_risk_raw(bayesian_severity: float, n_risk: int) -> float:
    """Объединённая метрика: байесовская серьёзность × √объём. Нормализуется к 1–5 по всем банкам."""
    if n_risk <= 0:
        return 0.0
    return bayesian_severity * math.sqrt(n_risk)


app = FastAPI(title="RiskGrabber")

settings = get_settings()


@app.on_event("startup")
def on_startup() -> None:
    settings = get_settings()
    log_level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    for noisy_logger in (
        "httpx",
        "httpcore",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "huggingface_hub.utils._http",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logger.info("Logging level set to %s (debug=%s)", logging.getLevelName(log_level), settings.debug)
    init_db()
    logger.info("Preloading embedder model: %s", settings.hf_model_name)
    try:
        embedder = get_embedder()
        test_embedding = embedder.embed(["test"])
        logger.info("Embedder model loaded successfully, embedding dimension: %s", len(test_embedding[0]))
    except Exception as e:
        logger.error("Failed to load embedder model: %s", e, exc_info=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/scrape/once")
async def scrape_once() -> dict:
    def _run() -> int:
        with SessionLocal() as db:
            return ingest_reviews(db, max_pages=settings.scrape_pages)
    count = await asyncio.to_thread(_run)
    return {"inserted": count}


@app.post("/scrape/backfill")
async def scrape_backfill(
    days: int = Query(default=7, ge=1, le=365, description="Количество последних дней для сбора отзывов"),
) -> dict:
    since_dt = moscow_now() - dt.timedelta(days=days)

    def _run() -> int:
        with SessionLocal() as db:
            return ingest_reviews_since(db, since=since_dt)
    count = await asyncio.to_thread(_run)
    return {"inserted": count, "days": days, "since": since_dt.isoformat()}


@app.post("/dataset/load-russian-bank-reviews")
async def dataset_load_russian_bank_reviews(
    limit: int = Query(default=50, ge=1, description="Сколько новых отзывов загрузить (отсортированы по дате: старые → новые). Без верхнего предела."),
) -> dict:
    def _run() -> dict:
        with SessionLocal() as db:
            return load_russian_bank_reviews_into_db(db, limit=limit)
    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.exception("Ошибка загрузки датасета: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/generate")
async def generate_embeddings(
    db: AsyncSession = Depends(get_db),
    sync_qdrant: bool = Query(
        default=False,
        description="Если true — после векторизации отправить все эмбеддинги в Qdrant",
    ),
) -> dict:
    embedded = await embed_new_reviews_async(db)
    synced_qdrant = 0
    if sync_qdrant:
        synced_qdrant = await sync_all_reviews_to_qdrant_async(db)
    return {
        "embedded_reviews": embedded,
        "qdrant_synced": synced_qdrant,
    }


@app.post("/pipeline/run")
async def run_pipeline(
    db: AsyncSession = Depends(get_db),
    sync_qdrant: bool = Query(
        default=False,
        description="Если true — после векторизации отправить все эмбеддинги в Qdrant",
    ),
) -> dict:
    embedded = await embed_new_reviews_async(db)
    synced_qdrant = 0
    if sync_qdrant:
        synced_qdrant = await sync_all_reviews_to_qdrant_async(db)
    return {
        "embedded_reviews": embedded,
        "qdrant_synced": synced_qdrant,
    }


@app.post("/qdrant/sync")
async def qdrant_sync(db: AsyncSession = Depends(get_db)) -> dict:
    synced = await sync_all_reviews_to_qdrant_async(db)
    return {"qdrant_synced": synced}


@app.get("/vectors/map", response_model=schemas.VectorMapResponse)
async def vectors_map(
    date_from: Optional[str] = Query(None, description="Начало периода по дате отзыва (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Конец периода по дате отзыва (YYYY-MM-DD)"),
    limit: int = Query(
        default=50000,
        ge=10,
        le=100000,
        description="Максимальное количество отзывов для визуализации (по умолчанию — все)",
    ),
    db: AsyncSession = Depends(get_db),
) -> schemas.VectorMapResponse:
    """
    Вернуть 2D-проекцию векторных представлений отзывов (UMAP, fallback на PCA).
    Период — по дате отзыва (date_from/date_to). Если не заданы — все отзывы с вектором.
    """
    review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
    since, until = None, None
    if date_from or date_to:
        try:
            if date_from:
                since = dt.datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=None)
            if date_to:
                until = dt.datetime.strptime(date_to, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59, microsecond=999999, tzinfo=None
                )
        except ValueError:
            pass
    stmt = (
        select(models.Review)
        .where(models.Review.vector_in_qdrant.is_(True))
    )
    if since is not None:
        stmt = stmt.where(review_date >= since)
    if until is not None:
        stmt = stmt.where(review_date <= until)
    stmt = stmt.order_by(review_date.desc()).limit(limit)
    result = await db.execute(stmt)
    reviews = result.scalars().all()
    if not reviews:
        return schemas.VectorMapResponse(points=[])

    review_ids = [r.id for r in reviews]
    id_to_vector = await get_vectors_by_review_ids_async(review_ids)
    if not id_to_vector:
        return schemas.VectorMapResponse(points=[])
    # Только отзывы, для которых вектор реально есть в Qdrant (могут расходиться с БД)
    reviews_with_vector = [r for r in reviews if r.id in id_to_vector]
    if not reviews_with_vector:
        return schemas.VectorMapResponse(points=[])
    X = np.array([id_to_vector[r.id] for r in reviews_with_vector], dtype=float)
    n_reviews = len(reviews_with_vector)

    # Одна точка — возвращаем координаты (0, 0)
    if n_reviews == 1:
        coords = np.array([[0.0, 0.0]], dtype=float)
    # Две и больше — UMAP или PCA
    elif umap is not None and n_reviews >= 2:
        try:
            logger.info("Using UMAP for 2D visualization of %s reviews", n_reviews)
            n_neighbors = min(15, max(2, n_reviews - 1))
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=2,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
                n_jobs=1,
            )
            coords = reducer.fit_transform(X)
        except Exception as e:
            logger.warning("UMAP failed (%s), falling back to PCA", e)
            pca = PCA(n_components=min(2, n_reviews))
            coords = pca.fit_transform(X)
    else:
        pca = PCA(n_components=min(2, n_reviews))
        coords = pca.fit_transform(X)

    points: list[schemas.VectorPoint] = []
    for review, (x, y) in zip(reviews_with_vector, coords):
        points.append(
            schemas.VectorPoint(
                review_id=review.id,
                bank_id=review.bank_id,
                x=float(x),
                y=float(y),
                rating=review.rating,
                sentiment=review.sentiment,
            )
        )

    return schemas.VectorMapResponse(points=points)


@app.get("/review/{review_id}", response_model=schemas.ReviewDetail)
async def get_review_detail(
    review_id: int,
    db: AsyncSession = Depends(get_db),
) -> schemas.ReviewDetail:
    result = await db.execute(select(models.Review).where(models.Review.id == review_id))
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    bank_r = await db.execute(select(models.Bank).where(models.Bank.id == review.bank_id))
    bank = bank_r.scalar_one_or_none()
    return schemas.ReviewDetail(
        id=review.id,
        bank=schemas.BankBase.model_validate(bank) if bank else None,
        rating=review.rating,
        title=review.title,
        text=review.text,
        published_at=review.published_at,
        scraped_at=review.scraped_at,
        sentiment=review.sentiment,
        sentiment_score=review.sentiment_score,
        criticality_score=review.criticality_score,
    )


@app.get("/risks/{risk_id}", response_model=schemas.RiskDetailWithReviews)
async def get_risk_detail(
    risk_id: int,
    db: AsyncSession = Depends(get_db),
) -> schemas.RiskDetailWithReviews:
    """Риск: вид, описание, риск-факторы и список привязанных отзывов."""
    risk_r = await db.execute(select(models.Risk).where(models.Risk.id == risk_id))
    risk = risk_r.scalar_one_or_none()
    if not risk:
        raise HTTPException(status_code=404, detail="Risk not found")
    links_r = await db.execute(select(models.ReviewRisk).where(models.ReviewRisk.risk_id == risk_id))
    links = links_r.scalars().all()
    review_ids = [lr.review_id for lr in links]
    reviews: List[models.Review] = []
    if review_ids:
        rev_r = await db.execute(select(models.Review).where(models.Review.id.in_(review_ids)))
        reviews = list(rev_r.scalars().all())
    bank_ids = {r.bank_id for r in reviews}
    banks_by_id: Dict[int, models.Bank] = {}
    if bank_ids:
        banks_r = await db.execute(select(models.Bank).where(models.Bank.id.in_(bank_ids)))
        banks_by_id = {b.id: b for b in banks_r.scalars().all()}
    review_details: List[schemas.ReviewDetail] = []
    for review in reviews:
        bank = banks_by_id.get(review.bank_id)
        review_details.append(
            schemas.ReviewDetail(
                id=review.id,
                bank=schemas.BankBase.model_validate(bank) if bank else None,
                rating=review.rating,
                title=review.title,
                text=review.text,
                published_at=review.published_at,
                scraped_at=review.scraped_at,
                sentiment=review.sentiment,
                sentiment_score=review.sentiment_score,
                criticality_score=review.criticality_score,
            )
        )
    return schemas.RiskDetailWithReviews(
        risk=schemas.RiskBase.model_validate(risk),
        reviews=review_details,
    )


@app.get("/risks/{risk_id}/reviews", response_model=List[schemas.ReviewDetail])
async def get_risk_reviews(
    risk_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[schemas.ReviewDetail]:
    """Список отзывов по риску (для обратной совместимости)."""
    data = await get_risk_detail(risk_id, db)
    return data.reviews


@app.post("/risks/generate")
async def generate_risks_endpoint(
    max_reviews: int = Query(
        default=10,
        ge=1,
        le=1000,
        description="Максимальное количество отрицательных отзывов для генерации рисков",
    ),
    db: AsyncSession = Depends(get_db),
) -> dict:
    from sqlalchemy import exists
    subq = select(models.ReviewRisk).where(models.ReviewRisk.review_id == models.Review.id)
    review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
    stmt = (
        select(models.Review)
        .where(
            models.Review.sentiment == "negative",
            models.Review.vector_in_qdrant.is_(True),
            ~exists(subq).correlate(models.Review),
        )
        .order_by(review_date.asc())
        .limit(max_reviews)
        .options(selectinload(models.Review.bank))
    )
    result = await db.execute(stmt)
    reviews = result.scalars().all()
    if not reviews:
        return {"reviews_processed": 0, "risks_created": 0}
    total_risks = 0
    for review in reviews:
        risk = await generate_risk_for_review_async(db, review)
        if risk:
            total_risks += 1
    return {
        "reviews_processed": len(reviews),
        "risks_created": total_risks,
    }


def _progress_queue(q: queue.Queue):
    """Возвращает progress_callback, который кладёт события в queue.Queue (для потоков)."""
    def progress(stage: str, message: str, detail: dict) -> None:
        q.put({"stage": stage, "message": message, **detail})
    return progress


def _progress_asyncio_queue(aq: asyncio.Queue):
    """Возвращает progress_callback для asyncio.Queue (тот же event loop, put_nowait)."""
    def progress(stage: str, message: str, detail: dict) -> None:
        aq.put_nowait({"stage": stage, "message": message, **detail})
    return progress


async def _stream_risks_run_async(aq: asyncio.Queue, max_reviews: int) -> None:
    """Генерация рисков с отправкой прогресса в asyncio.Queue (для работы в основном event loop)."""
    from sqlalchemy import exists
    async with AsyncSessionLocal() as db:
        subq = select(models.ReviewRisk).where(models.ReviewRisk.review_id == models.Review.id)
        review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
        stmt = (
            select(models.Review)
            .where(
                models.Review.sentiment == "negative",
                models.Review.vector_in_qdrant.is_(True),
                ~exists(subq).correlate(models.Review),
            )
            .order_by(review_date.asc())
            .limit(max_reviews)
            .options(selectinload(models.Review.bank))
        )
        result = await db.execute(stmt)
        reviews = result.scalars().all()
        if not reviews:
            msg = "Нет отзывов для генерации рисков. Нужны отзывы: сентимент «негативный», вектор в Qdrant, без присвоенного риска."
            await aq.put({
                "done": True,
                "message": msg,
                "result": {"reviews_processed": 0, "risks_created": 0, "message": msg},
            })
            return
        total_risks = 0
        for i, review in enumerate(reviews):
            await aq.put({
                "stage": "llm",
                "message": f"LLM: отзыв {i + 1} из {len(reviews)}...",
                "current": i + 1,
                "total": len(reviews),
                "risks_created": total_risks,
            })
            risk = await generate_risk_for_review_async(db, review)
            if risk:
                total_risks += 1
                await aq.put({
                    "stage": "llm",
                    "message": f"Создан риск. Всего: {total_risks}",
                    "risks_created": total_risks,
                })
        await aq.put({
            "done": True,
            "message": f"Готово. Создано рисков: {total_risks}, обработано отзывов: {len(reviews)}.",
            "result": {
                "reviews_processed": len(reviews),
                "risks_created": total_risks,
            },
        })


async def _stream_from_queue(q: queue.Queue):
    """Асинхронно читает из queue.Queue и отдаёт SSE строки."""
    while True:
        try:
            item = await asyncio.to_thread(q.get)
        except Exception:
            break
        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        if item.get("done"):
            break


async def _stream_from_asyncio_queue(aq: asyncio.Queue):
    """Читает из asyncio.Queue и отдаёт SSE строки (тот же event loop)."""
    while True:
        item = await aq.get()
        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        if item.get("done"):
            break


@app.get("/stream/embeddings/generate")
async def stream_embeddings_generate() -> StreamingResponse:
    """
    SSE: генерация эмбеддингов с потоком прогресса по батчам.
    Выполняется в основном event loop, чтобы AsyncSession/asyncpg работали корректно.
    """
    aq: asyncio.Queue = asyncio.Queue()

    async def run_and_feed():
        try:
            async with AsyncSessionLocal() as session:
                progress = _progress_asyncio_queue(aq)
                embedded = await embed_new_reviews_async(session, progress_callback=progress)
                await aq.put({"done": True, "result": {"embedded_reviews": embedded}})
        except Exception as e:
            logger.exception("Ошибка в stream/embeddings/generate: %s", e)
            await aq.put({"done": True, "result": {"embedded_reviews": 0}, "error": str(e)})

    asyncio.create_task(run_and_feed())
    return StreamingResponse(
        _stream_from_asyncio_queue(aq),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stream/pipeline/run")
async def stream_pipeline_run(
    sync_qdrant: bool = Query(default=False),
) -> StreamingResponse:
    """
    SSE: полный пайплайн (эмбеддинги + кластеризация) с потоком прогресса.
    Выполняется в основном event loop, чтобы AsyncSession/asyncpg работали корректно.
    """
    aq: asyncio.Queue = asyncio.Queue()

    async def run_and_feed():
        try:
            async with AsyncSessionLocal() as session:
                progress = _progress_asyncio_queue(aq)
                embedded = await embed_new_reviews_async(session, progress_callback=progress)
                synced_qdrant = 0
                if sync_qdrant:
                    progress("qdrant", "Синхронизация с Qdrant...", {})
                    synced_qdrant = await sync_all_reviews_to_qdrant_async(session)
                await aq.put({
                    "done": True,
                    "result": {
                        "embedded_reviews": embedded,
                        "qdrant_synced": synced_qdrant,
                    },
                })
        except Exception as e:
            logger.exception("Ошибка в stream/pipeline/run: %s", e)
            await aq.put({
                "done": True,
                "result": {"embedded_reviews": 0, "qdrant_synced": 0},
                "error": str(e),
            })

    asyncio.create_task(run_and_feed())
    return StreamingResponse(
        _stream_from_asyncio_queue(aq),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stream/risks/generate")
async def stream_risks_generate(
    max_reviews: int = Query(default=10, ge=1, le=1000),
) -> StreamingResponse:
    """
    SSE: генерация рисков через LLM с потоком прогресса по отзывам.
    Выполняется в основном event loop (без потока), чтобы AsyncSession/asyncpg работали корректно.
    """
    aq: asyncio.Queue = asyncio.Queue()
    await aq.put({"message": "Поиск отрицательных отзывов без риска...", "stage": "start"})

    async def run_and_feed():
        try:
            await _stream_risks_run_async(aq, max_reviews)
        except Exception as e:
            logger.exception("Ошибка в stream/risks/generate: %s", e)
            await aq.put({
                "done": True,
                "message": f"Ошибка: {e!s}",
                "result": {"reviews_processed": 0, "risks_created": 0},
            })

    asyncio.create_task(run_and_feed())
    return StreamingResponse(
        _stream_from_asyncio_queue(aq),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _parse_dashboard_dates(
    date_from: Optional[str],
    date_to: Optional[str],
    reviews_date_min: Optional[dt.datetime],
    reviews_date_max: Optional[dt.datetime],
):
    """Parse date_from/date_to and return (since, until) for analytics/dashboard."""
    since = reviews_date_min
    until = reviews_date_max
    if date_from or date_to:
        try:
            if date_from:
                since = dt.datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=None)
            if date_to:
                until = dt.datetime.strptime(date_to, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59, microsecond=999999, tzinfo=None
                )
        except ValueError:
            pass
    if since is None:
        since = moscow_now() - dt.timedelta(days=365)
    if until is None:
        until = moscow_now()
    if since > until:
        since, until = until, since
    return since, until


@app.get("/analytics/reviews-over-time", response_model=schemas.ReviewsOverTimeResponse)
def analytics_reviews_over_time(
    date_from: Optional[str] = Query(None, description="Начало периода (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Конец периода (YYYY-MM-DD)"),
    bank_id: Optional[int] = Query(None, description="Фильтр по банку (ID); не задан — все банки"),
    group_by: str = Query("day", description="Группировка: day, week, month"),
    db: Session = Depends(get_db_sync),
) -> schemas.ReviewsOverTimeResponse:
    """По дням/неделям/месяцам: отзывы (только позитивные и негативные) и количество уникальных рисков. Опционально по банку."""
    review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
    reviews_date_min = db.query(func.min(review_date)).scalar()
    reviews_date_max = db.query(func.max(review_date)).scalar()
    since, until = _parse_dashboard_dates(date_from, date_to, reviews_date_min, reviews_date_max)
    since_date = since.date() if since else dt.date.today()
    until_date = until.date() if until else dt.date.today()
    if since_date > until_date:
        since_date, until_date = until_date, since_date

    # Отзывы по дате и сентименту (только positive и negative)
    q = (
        db.query(
            func.date(review_date).label("d"),
            models.Review.sentiment,
            func.count(models.Review.id).label("cnt"),
        )
        .filter(review_date >= since, review_date <= until)
        .group_by(func.date(review_date), models.Review.sentiment)
    )
    if bank_id is not None:
        q = q.filter(models.Review.bank_id == bank_id)
    rows = q.all()
    by_date: Dict[dt.date, Dict[str, int]] = {}
    for d, sentiment, cnt in rows:
        day = d if hasattr(d, "year") else (dt.datetime.fromisoformat(str(d)).date() if isinstance(d, str) else d)
        if day not in by_date:
            by_date[day] = {"positive": 0, "negative": 0}
        if sentiment == "positive":
            by_date[day]["positive"] = int(cnt or 0)
        elif sentiment == "negative":
            by_date[day]["negative"] = int(cnt or 0)

    # Уникальные риски по дням (count distinct risk_id)
    risk_date = func.coalesce(models.ReviewRisk.review_date, models.ReviewRisk.created_at)
    rq = (
        db.query(
            func.date(risk_date).label("d"),
            func.count(distinct(models.ReviewRisk.risk_id)).label("cnt"),
        )
        .filter(risk_date >= since, risk_date <= until)
        .group_by(func.date(risk_date))
    )
    if bank_id is not None:
        rq = rq.filter(models.ReviewRisk.bank_id == bank_id)
    risk_rows = rq.all()
    risks_by_date: Dict[dt.date, int] = {}
    for d, cnt in risk_rows:
        day = d if hasattr(d, "year") else (dt.datetime.fromisoformat(str(d)).date() if isinstance(d, str) else d)
        risks_by_date[day] = int(cnt or 0)

    # Собираем интервалы в зависимости от group_by
    if group_by == "month":
        # Агрегируем по месяцу: ключ (year, month)
        agg: Dict[tuple, Dict] = {}
        current = since_date
        while current <= until_date:
            key = (current.year, current.month)
            if key not in agg:
                agg[key] = {"positive": 0, "negative": 0, "risks": 0, "start": current}
            rec = by_date.get(current, {"positive": 0, "negative": 0})
            agg[key]["positive"] += rec["positive"]
            agg[key]["negative"] += rec["negative"]
            agg[key]["risks"] += risks_by_date.get(current, 0)
            current += dt.timedelta(days=1)
        buckets = [
            schemas.ReviewsOverTimeBucket(
                date=dt.date(y, m, 1).isoformat(),
                positive=data["positive"],
                negative=data["negative"],
                risks_count=data["risks"],
            )
            for (y, m), data in sorted(agg.items())
        ]
    elif group_by == "week":
        # Агрегируем по неделе (понедельник — начало)
        agg: Dict[dt.date, Dict] = {}
        current = since_date
        while current <= until_date:
            wd = current.weekday()
            week_start = current - dt.timedelta(days=wd)
            if week_start not in agg:
                agg[week_start] = {"positive": 0, "negative": 0, "risks": 0}
            rec = by_date.get(current, {"positive": 0, "negative": 0})
            agg[week_start]["positive"] += rec["positive"]
            agg[week_start]["negative"] += rec["negative"]
            agg[week_start]["risks"] += risks_by_date.get(current, 0)
            current += dt.timedelta(days=1)
        buckets = [
            schemas.ReviewsOverTimeBucket(
                date=week_start.isoformat(),
                positive=data["positive"],
                negative=data["negative"],
                risks_count=data["risks"],
            )
            for week_start, data in sorted(agg.items())
        ]
    else:
        # day
        buckets = []
        current = since_date
        while current <= until_date:
            rec = by_date.get(current, {"positive": 0, "negative": 0})
            buckets.append(
                schemas.ReviewsOverTimeBucket(
                    date=current.isoformat(),
                    positive=rec["positive"],
                    negative=rec["negative"],
                    risks_count=risks_by_date.get(current, 0),
                )
            )
            current += dt.timedelta(days=1)
    return schemas.ReviewsOverTimeResponse(buckets=buckets)


def _bucket_start(d: dt.datetime, group_by: str) -> dt.datetime:
    """Return start of day, week or month for grouping."""
    if group_by == "month":
        return d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if group_by == "week":
        # Monday as week start
        weekday = d.weekday()
        delta = dt.timedelta(days=weekday)
        return (d - delta).replace(hour=0, minute=0, second=0, microsecond=0)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


@app.get("/analytics/risk-trends", response_model=schemas.RiskTrendsResponse)
def analytics_risk_trends(
    date_from: Optional[str] = Query(None, description="Начало периода (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Конец периода (YYYY-MM-DD)"),
    bank_id: Optional[int] = Query(None, description="Фильтр по банку (ID); не задан — все банки"),
    group_by: str = Query("week", description="Интервал: day, week, month"),
    risk_type: Optional[str] = Query(None, description="Фильтр по типу риска"),
    limit_risks: Optional[int] = Query(15, ge=1, le=50, description="Топ N рисков по количеству"),
    db: Session = Depends(get_db_sync),
) -> schemas.RiskTrendsResponse:
    """Тренды по рискам: количество отзывов по видам рисков по временным интервалам. Опционально по банку."""
    review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
    reviews_date_min = db.query(func.min(review_date)).scalar()
    reviews_date_max = db.query(func.max(review_date)).scalar()
    risk_date = func.coalesce(models.ReviewRisk.review_date, models.ReviewRisk.created_at)
    since, until = _parse_dashboard_dates(date_from, date_to, reviews_date_min, reviews_date_max)

    q = (
        db.query(
            models.ReviewRisk,
            models.Risk,
        )
        .join(models.Risk, models.ReviewRisk.risk_id == models.Risk.id)
        .filter(risk_date >= since, risk_date <= until)
    )
    if bank_id is not None:
        q = q.filter(models.ReviewRisk.bank_id == bank_id)
    if risk_type:
        q = q.filter(models.Risk.risk_type == risk_type)
    rows = q.all()

    # Bucket by day, week or month
    bucket_key = "month" if group_by == "month" else ("week" if group_by == "week" else "day")
    by_bucket: Dict[dt.datetime, Dict[int, int]] = {}  # bucket_start -> { risk_id: count }
    risk_meta_by_id: Dict[int, models.Risk] = {}
    for rr, risk in rows:
        risk_meta_by_id[risk.id] = risk
        t = rr.review_date or rr.created_at
        if t is None:
            continue
        start = _bucket_start(t, bucket_key)
        by_bucket.setdefault(start, {})
        by_bucket[start][risk.id] = by_bucket[start].get(risk.id, 0) + 1

    # Top risks by total count
    total_by_risk: Dict[int, int] = {}
    for buck in by_bucket.values():
        for rid, c in buck.items():
            total_by_risk[rid] = total_by_risk.get(rid, 0) + c
    top_risk_ids = sorted(total_by_risk.keys(), key=lambda x: -total_by_risk[x])[: limit_risks or 15]

    intervals: List[schemas.RiskTrendBucket] = []
    for start in sorted(by_bucket.keys()):
        if bucket_key == "month":
            # first day of next month
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        elif bucket_key == "week":
            end = start + dt.timedelta(days=7)
        else:
            end = start + dt.timedelta(days=1)
        risks_in_bucket = [
            schemas.RiskCountInBucket(
                risk_id=rid,
                risk_type=risk_meta_by_id[rid].risk_type,
                count=by_bucket[start].get(rid, 0),
            )
            for rid in top_risk_ids
            if rid in risk_meta_by_id
        ]
        intervals.append(
            schemas.RiskTrendBucket(start=start, end=end, risks=risks_in_bucket)
        )

    risk_meta = [schemas.RiskBase.model_validate(risk_meta_by_id[rid]) for rid in top_risk_ids if rid in risk_meta_by_id]
    return schemas.RiskTrendsResponse(intervals=intervals, risk_meta=risk_meta)


@app.get("/analytics/bank-risk-trends", response_model=schemas.BankRiskTrendsResponse)
def analytics_bank_risk_trends(
    bank_id: int = Query(..., description="ID банка"),
    date_from: Optional[str] = Query(None, description="Начало периода (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Конец периода (YYYY-MM-DD)"),
    group_by: str = Query("day", description="Интервал: day или week"),
    db: Session = Depends(get_db_sync),
) -> schemas.BankRiskTrendsResponse:
    """Тренды по рискам по выбранному банку."""
    bank = db.query(models.Bank).filter(models.Bank.id == bank_id).first()
    if not bank:
        raise HTTPException(status_code=404, detail="Bank not found")
    review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
    reviews_date_min = db.query(func.min(review_date)).scalar()
    reviews_date_max = db.query(func.max(review_date)).scalar()
    risk_date = func.coalesce(models.ReviewRisk.review_date, models.ReviewRisk.created_at)
    since, until = _parse_dashboard_dates(date_from, date_to, reviews_date_min, reviews_date_max)

    rows = (
        db.query(models.ReviewRisk, models.Risk)
        .join(models.Risk, models.ReviewRisk.risk_id == models.Risk.id)
        .filter(
            models.ReviewRisk.bank_id == bank_id,
            risk_date >= since,
            risk_date <= until,
        )
        .all()
    )

    bucket_key = "week" if group_by == "week" else "day"
    by_bucket: Dict[dt.datetime, Dict[int, int]] = {}
    risk_meta_by_id: Dict[int, models.Risk] = {}
    for rr, risk in rows:
        risk_meta_by_id[risk.id] = risk
        t = rr.review_date or rr.created_at
        if t is None:
            continue
        start = _bucket_start(t, bucket_key)
        by_bucket.setdefault(start, {})
        by_bucket[start][risk.id] = by_bucket[start].get(risk.id, 0) + 1

    intervals = []
    all_risk_ids = sorted(set().union(*(b.keys() for b in by_bucket.values())))
    for start in sorted(by_bucket.keys()):
        end = start + (dt.timedelta(days=7) if bucket_key == "week" else dt.timedelta(days=1))
        risks_in_bucket = [
            schemas.RiskCountInBucket(
                risk_id=rid,
                risk_type=risk_meta_by_id[rid].risk_type,
                count=by_bucket[start].get(rid, 0),
            )
            for rid in all_risk_ids
        ]
        intervals.append(schemas.RiskTrendBucket(start=start, end=end, risks=risks_in_bucket))

    risk_meta = [schemas.RiskBase.model_validate(risk_meta_by_id[rid]) for rid in all_risk_ids]
    return schemas.BankRiskTrendsResponse(
        bank_id=bank.id,
        bank_name=bank.name,
        intervals=intervals,
        risk_meta=risk_meta,
    )


@app.get("/analytics/risk-matrix", response_model=schemas.RiskMatrixResponse)
async def analytics_risk_matrix(
    date_from: Optional[str] = Query(None, description="Начало периода (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Конец периода (YYYY-MM-DD)"),
    bank_id: Optional[int] = Query(None, description="Фильтр по банку"),
    limit_risks: Optional[int] = Query(30, ge=1, le=100, description="Топ N рисков"),
    db: AsyncSession = Depends(get_db),
) -> schemas.RiskMatrixResponse:
    """Матрица банк × вид риска: средняя серьёзность отзывов (1–5) для каждой пары."""
    review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
    r_min = await db.execute(select(func.min(review_date)).select_from(models.Review))
    reviews_date_min = r_min.scalar_one_or_none()
    r_max = await db.execute(select(func.max(review_date)).select_from(models.Review))
    reviews_date_max = r_max.scalar_one_or_none()
    risk_date = func.coalesce(models.ReviewRisk.review_date, models.ReviewRisk.created_at)
    since, until = _parse_dashboard_dates(date_from, date_to, reviews_date_min, reviews_date_max)

    stmt = (
        select(
            models.ReviewRisk.bank_id,
            models.Risk.risk_type,
            func.avg(models.Review.criticality_score).label("avg_criticality"),
            func.count(models.ReviewRisk.id).label("cnt"),
        )
        .join(models.Review, models.ReviewRisk.review_id == models.Review.id)
        .join(models.Risk, models.ReviewRisk.risk_id == models.Risk.id)
        .where(risk_date >= since, risk_date <= until)
        .group_by(models.ReviewRisk.bank_id, models.Risk.risk_type)
    )
    if bank_id is not None:
        stmt = stmt.where(models.ReviewRisk.bank_id == bank_id)
    result = await db.execute(stmt)
    rows = result.all()

    bank_ids = list({r[0] for r in rows})
    bank_by_id = {}
    if bank_ids:
        r_banks = await db.execute(select(models.Bank).where(models.Bank.id.in_(bank_ids)))
        bank_by_id = {b.id: b for b in r_banks.scalars().all()}

    # Топ видов риска по количеству связок (для порядка столбцов)
    type_totals = {}
    for bid, rtype, avg_crit, cnt in rows:
        type_totals[rtype] = type_totals.get(rtype, 0) + int(cnt or 0)
    risk_types = sorted(type_totals.keys(), key=lambda x: -type_totals[x])[: limit_risks or 30]

    cells = []
    for (bid, rtype, avg_criticality, cnt) in rows:
        if rtype not in risk_types:
            continue
        bank = bank_by_id.get(bid)
        if not bank:
            continue
        avg_val = float(avg_criticality) if avg_criticality is not None else None
        cells.append(
            schemas.RiskMatrixCell(
                bank_id=bid,
                bank_name=bank.name,
                risk_type=rtype,
                avg_criticality=round(avg_val, 1) if avg_val is not None else None,
            )
        )

    return schemas.RiskMatrixResponse(
        cells=cells,
        banks=[schemas.BankBase.model_validate(b) for b in bank_by_id.values()],
        risk_types=risk_types,
    )


@app.get("/analytics/bank-scores", response_model=List[schemas.BankScoreItem])
def analytics_bank_scores(
    date_from: Optional[str] = Query(None, description="Начало периода (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Конец периода (YYYY-MM-DD)"),
    db: Session = Depends(get_db_sync),
) -> List[schemas.BankScoreItem]:
    """Рейтинг банков по индексу рисковости: средняя критичность отзывов (1–5) за период по отзывам с рисками."""
    review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
    reviews_date_min = db.query(func.min(review_date)).scalar()
    reviews_date_max = db.query(func.max(review_date)).scalar()
    risk_date = func.coalesce(models.ReviewRisk.review_date, models.ReviewRisk.created_at)
    since, until = _parse_dashboard_dates(date_from, date_to, reviews_date_min, reviews_date_max)

    # Per-bank: sum of review criticality_score (1–5) for reviews with risks in period
    rr_rows = (
        db.query(models.ReviewRisk.bank_id, models.Review.criticality_score)
        .join(models.Review, models.ReviewRisk.review_id == models.Review.id)
        .filter(
            risk_date >= since,
            risk_date <= until,
            models.Review.criticality_score.isnot(None),
        )
        .all()
    )
    weighted_by_bank: Dict[int, float] = {}
    for bid, score in rr_rows:
        s = int(score) if score is not None else 0
        weighted_by_bank[bid] = weighted_by_bank.get(bid, 0) + s

    review_counts = (
        db.query(models.Review.bank_id, func.count(models.Review.id))
        .filter(review_date >= since, review_date <= until)
        .group_by(models.Review.bank_id)
        .all()
    )
    reviews_by_bank = {bid: int(c) for bid, c in review_counts}

    risk_counts = (
        db.query(models.ReviewRisk.bank_id, func.count(models.ReviewRisk.id))
        .filter(risk_date >= since, risk_date <= until)
        .group_by(models.ReviewRisk.bank_id)
        .all()
    )
    risks_by_bank = {bid: int(c) for bid, c in risk_counts}

    bank_ids = set(weighted_by_bank.keys()) | set(reviews_by_bank.keys())
    banks = db.query(models.Bank).filter(models.Bank.id.in_(bank_ids)).all() if bank_ids else []
    # Объединённая метрика: байес × √объём, нормализация к 1–5; средняя серьёзность w/n_risk
    rows_data: List[tuple] = []
    unified_raws: List[float] = []
    for bank in banks:
        bid = bank.id
        w = weighted_by_bank.get(bid, 0)
        rev_count = reviews_by_bank.get(bid, 0)
        risk_count = risks_by_bank.get(bid, 0)
        risk_score_bay = _risk_score_bayesian(w, risk_count)
        avg_sev = round(w / risk_count, 4) if risk_count else 0.0
        raw_unified = _unified_risk_raw(risk_score_bay, risk_count)
        unified_raws.append(raw_unified)
        rows_data.append((bank, rev_count, risk_count, risk_score_bay, avg_sev, raw_unified))
    max_raw = max(unified_raws) if unified_raws else 0.0
    result: List[schemas.BankScoreItem] = []
    for bank, rev_count, risk_count, _bay, avg_sev, raw_unified in rows_data:
        if max_raw > 0 and raw_unified >= 0:
            risk_score_val = 1.0 + 4.0 * (raw_unified / max_raw)
            risk_score_val = max(1.0, min(5.0, round(risk_score_val, 4)))
        else:
            risk_score_val = 0.0
        result.append(
            schemas.BankScoreItem(
                bank_id=bank.id,
                bank_name=bank.name,
                risk_score=risk_score_val,
                risks_count=risk_count,
                total_reviews=rev_count,
                avg_severity=avg_sev if risk_count else None,
            )
        )
    result.sort(key=lambda x: -x.risk_score)
    return result


@app.get("/analytics/review-criticality", response_model=schemas.ReviewCriticalityResponse)
async def analytics_review_criticality(
    date_from: Optional[str] = Query(None, description="Начало периода (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Конец периода (YYYY-MM-DD)"),
    bank_id: Optional[int] = Query(None, description="Фильтр по банку"),
    db: AsyncSession = Depends(get_db),
) -> schemas.ReviewCriticalityResponse:
    """Распределение отзывов по степени критичности (1–5) по компаниям (банкам)."""
    review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
    r_min = await db.execute(select(func.min(review_date)))
    r_max = await db.execute(select(func.max(review_date)))
    reviews_date_min = r_min.scalar()
    reviews_date_max = r_max.scalar()
    since, until = _parse_dashboard_dates(date_from, date_to, reviews_date_min, reviews_date_max)

    stmt = (
        select(models.Review.bank_id, models.Review.criticality_score, func.count(models.Review.id))
        .where(
            review_date >= since,
            review_date <= until,
            models.Review.criticality_score.isnot(None),
        )
        .group_by(models.Review.bank_id, models.Review.criticality_score)
    )
    if bank_id is not None:
        stmt = stmt.where(models.Review.bank_id == bank_id)
    result = await db.execute(stmt)
    rows = result.all()

    by_bank: Dict[int, Dict[int, int]] = {}
    for bid, score, cnt in rows:
        if bid not in by_bank:
            by_bank[bid] = {}
        by_bank[bid][int(score)] = int(cnt or 0)

    bank_ids = list(by_bank.keys())
    banks: List[models.Bank] = []
    if bank_ids:
        res = await db.execute(select(models.Bank).where(models.Bank.id.in_(bank_ids)))
        banks = list(res.scalars().all())
    bank_by_id = {b.id: b for b in banks}

    result_banks: List[schemas.BankCriticalityItem] = []
    for bid in sorted(by_bank.keys()):
        dist = by_bank[bid]
        distribution = [
            schemas.CriticalityCount(score=s, count=dist.get(s, 0))
            for s in range(1, 6)
        ]
        bank_name = getattr(bank_by_id.get(bid), "name", None) or str(bid)
        result_banks.append(
            schemas.BankCriticalityItem(
                bank_id=bid,
                bank_name=bank_name,
                distribution=distribution,
            )
        )
    return schemas.ReviewCriticalityResponse(banks=result_banks)


@app.get("/dashboard/data", response_model=schemas.DashboardResponse)
def dashboard_data(
    bank_id: Optional[str] = Query(None, description="Фильтр рисков по банку (ID)"),
    date_from: Optional[str] = Query(None, description="Начало периода по дате отзыва (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Конец периода по дате отзыва (YYYY-MM-DD)"),
    db: Session = Depends(get_db_sync),
) -> schemas.DashboardResponse:
    """
    Статистика дашборда. Период — по дате отзыва (published_at / scraped_at).
    Если date_from/date_to не заданы, берётся весь диапазон (мин/макс даты отзывов в БД).
    """
    bank_id_int: Optional[int] = None
    if bank_id is not None and str(bank_id).strip():
        try:
            bank_id_int = int(bank_id)
        except (TypeError, ValueError):
            pass

    review_date = func.coalesce(models.Review.published_at, models.Review.scraped_at)
    reviews_date_min = db.query(func.min(review_date)).scalar()
    reviews_date_max = db.query(func.max(review_date)).scalar()

    since = reviews_date_min
    until = reviews_date_max
    if date_from or date_to:
        try:
            if date_from:
                since = dt.datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=None)
            if date_to:
                until = dt.datetime.strptime(date_to, "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59, microsecond=999999, tzinfo=None
                )
        except ValueError:
            pass
    if since is None:
        since = moscow_now() - dt.timedelta(days=365)
    if until is None:
        until = moscow_now()
    if since > until:
        since, until = until, since

    risk_date = func.coalesce(models.ReviewRisk.review_date, models.ReviewRisk.created_at)

    # Per-bank stats (в периоде по дате отзыва), только позитивные и негативные
    pos_count = func.count(case((models.Review.sentiment == "positive", 1)))
    neg_count = func.count(case((models.Review.sentiment == "negative", 1)))
    bank_rows = (
        db.query(
            models.Bank,
            func.count(models.Review.id).label("total_reviews"),
            pos_count.label("positive_reviews"),
            neg_count.label("negative_reviews"),
            func.avg(models.Review.rating).label("avg_rating"),
            func.max(review_date).label("last_review_at"),
            func.count(models.ReviewRisk.id.distinct()).label("risks_count"),
        )
        .outerjoin(models.Review, models.Review.bank_id == models.Bank.id)
        .outerjoin(models.ReviewRisk, models.ReviewRisk.review_id == models.Review.id)
        .filter(review_date >= since, review_date <= until)
        .group_by(models.Bank.id)
        .all()
    )

    # Bank risk scores: средняя критичность отзывов (1–5) за период
    rr_weight_rows = (
        db.query(models.ReviewRisk.bank_id, models.Review.criticality_score)
        .join(models.Review, models.ReviewRisk.review_id == models.Review.id)
        .filter(
            risk_date >= since,
            risk_date <= until,
            models.Review.criticality_score.isnot(None),
        )
        .all()
    )
    weighted_by_bank: Dict[int, float] = {}
    for bid, score in rr_weight_rows:
        s = int(score) if score is not None else 0
        weighted_by_bank[bid] = weighted_by_bank.get(bid, 0) + s
    risk_count_by_bank_q = (
        db.query(models.ReviewRisk.bank_id, func.count(models.ReviewRisk.id))
        .filter(risk_date >= since, risk_date <= until)
        .group_by(models.ReviewRisk.bank_id)
        .all()
    )
    risk_count_by_bank = {bid: int(c) for bid, c in risk_count_by_bank_q}

    # Объединённая метрика: байес × √объём, нормализация к 1–5; средняя серьёзность
    score_data_by_bank: Dict[int, tuple] = {}
    unified_raws_dash: List[float] = []
    for row in bank_rows:
        bank = row[0]
        risks_count = int(row[6] or 0)
        w = weighted_by_bank.get(bank.id, 0)
        risk_score_bay = _risk_score_bayesian(w, risks_count)
        avg_sev = round(w / risks_count, 4) if risks_count else 0.0
        raw_unified = _unified_risk_raw(risk_score_bay, risks_count)
        unified_raws_dash.append(raw_unified)
        score_data_by_bank[bank.id] = (risk_score_bay, avg_sev, raw_unified)
    max_raw_dash = max(unified_raws_dash) if unified_raws_dash else 0.0

    banks: List[schemas.BankStats] = []
    bank_scores_list: List[schemas.BankScoreItem] = []
    for row in bank_rows:
        bank = row[0]
        total_reviews = int(row[1] or 0)
        positive_reviews = int(row[2] or 0)
        negative_reviews = int(row[3] or 0)
        avg_rating = row[4]
        last_review_at = row[5]
        risks_count = int(row[6] or 0)
        _bay, avg_sev, raw_unified = score_data_by_bank.get(
            bank.id, (0.0, 0.0, 0.0)
        )
        if max_raw_dash > 0 and raw_unified >= 0:
            risk_score_val = 1.0 + 4.0 * (raw_unified / max_raw_dash)
            risk_score_val = max(1.0, min(5.0, round(risk_score_val, 4)))
        else:
            risk_score_val = 0.0
        bank_scores_list.append(
            schemas.BankScoreItem(
                bank_id=bank.id,
                bank_name=bank.name,
                risk_score=risk_score_val,
                risks_count=risk_count_by_bank.get(bank.id, 0),
                total_reviews=total_reviews,
                avg_severity=avg_sev if risks_count else None,
            )
        )
        bank_data = schemas.BankBase(
            id=bank.id,
            name=bank.name,
            slug=bank.slug,
        )
        banks.append(
            schemas.BankStats(
                bank=bank_data,
                total_reviews=total_reviews,
                positive_reviews=positive_reviews,
                negative_reviews=negative_reviews,
                avg_rating=float(avg_rating) if avg_rating is not None else None,
                last_review_at=last_review_at,
                risks_count=risks_count,
                risk_score=risk_score_val if risk_score_val is not None else 0.0,
            )
        )
    bank_scores_list.sort(key=lambda x: -x.risk_score)

    # Overall time-bucketed stats (by day) — по дате отзыва
    bucket_rows = (
        db.query(
            func.date(review_date).label("bucket"),
            func.count(models.Review.id).label("reviews_count"),
            func.avg(models.Review.rating).label("avg_rating"),
        )
        .filter(review_date >= since, review_date <= until)
        .group_by(func.date(review_date))
        .order_by(func.date(review_date))
        .all()
    )

    # Risks count per day (by review_risks.review_date / created_at) for same period
    risk_bucket_rows = (
        db.query(
            func.date(risk_date).label("bucket"),
            func.count(models.ReviewRisk.id).label("risks_count"),
        )
        .filter(risk_date >= since, risk_date <= until)
        .group_by(func.date(risk_date))
        .all()
    )
    risks_per_bucket: Dict[dt.date, int] = {}
    for rb, rc in risk_bucket_rows:
        d = rb if hasattr(rb, "year") else dt.datetime.fromisoformat(str(rb)).date() if isinstance(rb, str) else rb
        risks_per_bucket[d] = int(rc or 0)

    buckets: List[schemas.TimeBucketStats] = []
    for bucket, reviews_count, avg_rating in bucket_rows:
        if isinstance(bucket, str):
            start = dt.datetime.fromisoformat(bucket)
        else:
            start = dt.datetime.combine(bucket, dt.time.min)
        bucket_date = start.date()
        end = start + dt.timedelta(days=1)
        buckets.append(
            schemas.TimeBucketStats(
                start=start,
                end=end,
                reviews_count=int(reviews_count or 0),
                avg_rating=float(avg_rating) if avg_rating is not None else None,
                risks_count=risks_per_bucket.get(bucket_date, 0),
            )
        )

    # Sentiment stats — по дате отзыва
    sentiment_rows = (
        db.query(models.Review.sentiment, func.count(models.Review.id))
        .filter(review_date >= since, review_date <= until)
        .group_by(models.Review.sentiment)
        .all()
    )
    pos = neg = 0
    for label, count in sentiment_rows:
        if label == "positive":
            pos = int(count or 0)
        elif label == "negative":
            neg = int(count or 0)

    sentiment = schemas.SentimentStats(positive=pos, negative=neg)

    # Период для рисков — тот же, что и для отзывов (since/until)
    risk_since = since
    risk_until = until

    # Актуальные риски: фильтр по дате отзыва (review_date), при отсутствии — по created_at
    risk_date = func.coalesce(models.ReviewRisk.review_date, models.ReviewRisk.created_at)
    rr_subq = (
        db.query(models.ReviewRisk)
        .filter(
            risk_date >= risk_since,
            risk_date <= risk_until,
        )
    )
    if bank_id_int is not None:
        rr_subq = rr_subq.filter(models.ReviewRisk.bank_id == bank_id_int)
    rr_ids = [r.id for r in rr_subq.all()]
    if not rr_ids:
        risks = []
    else:
        rr_rows = (
            db.query(models.ReviewRisk, models.Risk, models.Bank)
            .join(models.Risk, models.ReviewRisk.risk_id == models.Risk.id)
            .outerjoin(models.Bank, models.ReviewRisk.bank_id == models.Bank.id)
            .filter(models.ReviewRisk.id.in_(rr_ids))
            .all()
        )
        # Группируем по risk_id
        by_risk: dict[int, list] = {}
        for rr, risk, bank in rr_rows:
            by_risk.setdefault(risk.id, []).append((rr, risk, bank))
        risks = []
        for rid, items in by_risk.items():
            rr, risk, bank = items[0]
            bank_schema = schemas.BankBase(id=bank.id, name=bank.name, slug=bank.slug) if bank else None
            created_at = min(
                (x[0].review_date or x[0].created_at for x in items),
                default=rr.review_date or rr.created_at,
            )
            risks.append(
                schemas.RiskSummary(
                    id=risk.id,
                    bank=bank_schema,
                    risk_type=risk.risk_type,
                    description=risk.description or "",
                    created_at=created_at,
                    reviews_count=len(items),
                )
            )
        risks.sort(key=lambda r: r.created_at, reverse=True)

    # Общая статистика (всего по БД)
    total_reviews = db.query(func.count(models.Review.id)).scalar() or 0
    total_banks = db.query(func.count(models.Bank.id)).scalar() or 0

    # Мин/макс даты отзывов в БД (уже вычислены выше)

    # Обработано / не обработано сентиментом
    reviews_with_sentiment = (
        db.query(func.count(models.Review.id)).filter(models.Review.sentiment.isnot(None)).scalar() or 0
    )
    reviews_without_sentiment = (
        db.query(func.count(models.Review.id)).filter(models.Review.sentiment.is_(None)).scalar() or 0
    )

    # Отрицательные отзывы без присвоенного риска
    negative_reviews_without_risk = (
        db.query(func.count(models.Review.id))
        .filter(
            models.Review.sentiment == "negative",
            ~db.query(models.ReviewRisk).filter(models.ReviewRisk.review_id == models.Review.id).exists()
        )
        .scalar() or 0
    )
    # Отрицательные отзывы с риском
    negative_reviews_with_risk = (
        db.query(func.count(models.Review.id))
        .filter(
            models.Review.sentiment == "negative",
            db.query(models.ReviewRisk).filter(models.ReviewRisk.review_id == models.Review.id).exists()
        )
        .scalar() or 0
    )

    return schemas.DashboardResponse(
        banks=banks,
        overall=buckets,
        sentiment=sentiment,
        risks=risks,
        total_reviews=int(total_reviews),
        total_banks=int(total_banks),
        reviews_date_min=reviews_date_min,
        reviews_date_max=reviews_date_max,
        reviews_with_sentiment=int(reviews_with_sentiment),
        reviews_without_sentiment=int(reviews_without_sentiment),
        negative_reviews_without_risk=int(negative_reviews_without_risk),
        negative_reviews_with_risk=int(negative_reviews_with_risk),
        bank_scores=bank_scores_list,
    )


@app.get("/", response_class=HTMLResponse)
def dashboard_page() -> str:
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return frontend_path.read_text(encoding="utf-8")
