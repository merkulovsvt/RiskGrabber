import asyncio
import json
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

from ..backend.config import get_settings
from .utils import safe_prompt_str
from .embeddings import (
    get_embedder,
    format_risk_for_embed,
    INSTRUCT_RISK_SEARCH_PROMPT,
)
from .vector_store import (
    get_vector_for_review_async,
    search_risks_by_vector_async,
    upsert_risk_vector,
)
from .prompts import (
    GENERATOR_SYSTEM,
    CRITIC_SYSTEM,
    GENERATOR_USER_TEMPLATE,
    CRITIC_USER_TEMPLATE,
    GeneratorRisk,
    CriticResponse,
    DEDUB_SYSTEM,
    DEDUB_USER_TEMPLATE,
    DedubResponse,
    CRITICALITY_SYSTEM,
    CRITICALITY_USER_TEMPLATE,
    ReviewCriticalityResponse,
)
from ..backend import models

logger = logging.getLogger(__name__)

settings = get_settings()


class PipelineState(TypedDict, total=False):
    review_text: str
    review_title: Optional[str]
    bank_name: str
    known_risks: List[Dict[str, Any]]
    review_id: int
    bank_id: int
    current_risk: Optional[Dict[str, Any]]
    critic_iter: int
    critic_comment: str
    final_decision: str
    resolved_risk_id: Optional[int]
    is_new_risk: bool
    description: str
    risk_factors: Optional[List[str]]
    implications: Optional[List[str]]


llm = ChatOpenAI(
    model=settings.llm_model,
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    temperature=0.7,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    max_retries=2,
    timeout=10
)

structured_generator_llm = llm.with_structured_output(GeneratorRisk)
structured_critic_llm = llm.with_structured_output(CriticResponse)
structured_dedub_llm = llm.with_structured_output(DedubResponse)
structured_criticality_llm = llm.with_structured_output(ReviewCriticalityResponse)


async def generator_agent(state: PipelineState) -> PipelineState:
    review_id = state.get("review_id")
    critic_iter = state.get("critic_iter", 0)
    logger.info("Отзыв %s — Генератор: заход %s", review_id, critic_iter + 1)

    review_text = state["review_text"]
    review_title = state.get("review_title") or "Без темы"
    bank_name = state["bank_name"]
    critic_comment = state.get("critic_comment") or ""

    review_title_s = (review_title or "").strip()
    review_text_s = (review_text or "").strip()
    incident_text = f"{review_title_s}\n\n{review_text_s}"
    risks_to_regenerate = (critic_comment or "Пусто. Первичная генерация по инциденту.").strip()

    user_content = GENERATOR_USER_TEMPLATE.format(
        company_report=f"Банк {(bank_name or '').strip()}",
        incident_text=incident_text,
        risks_to_regenerate=risks_to_regenerate,
    )
    messages = [
        SystemMessage(content=GENERATOR_SYSTEM),
        HumanMessage(content=user_content),
    ]
    try:
        result: GeneratorRisk = await structured_generator_llm.ainvoke(messages)
        if not (result.description or result.risk_type):
            logger.info("Отзыв %s — Генератор: риск не сгенерирован (пустой ответ)", review_id)
            if critic_iter == 0:
                state["current_risk"] = None
            return state

        state["current_risk"] = {
            "risk_id": None,
            "risk_type": result.risk_type,
            "description": result.description,
            "risk_factors": result.risk_factor,
            "implications": result.implications,
        }
        logger.info(
            "Отзыв %s — Генератор: получен риск, вид=%s, длина описания=%s",
            review_id, (result.risk_type or "")[:50], len(result.description or ""),
        )
    except Exception as e:
        logger.warning("Ошибка генератора риска: %s", e, exc_info=True)

        if critic_iter == 0:
            state["current_risk"] = None
        else:
            logger.info("Отзыв %s — Генератор: при сбое оставляем предыдущий риск для дедубликации", review_id)
    return state


async def critic_agent(state: PipelineState) -> PipelineState:
    review_id = state.get("review_id")
    logger.info("Отзыв %s — Критик: проверка сгенерированного риска", review_id)

    critic_iter = state.get("critic_iter", 0)
    max_iter = get_settings().llm_critic_max_iter
    current_risk = state.get("current_risk")
    if not current_risk:
        logger.info("Отзыв %s — Критик: риска нет, отправка в дедубликацию", review_id)
        state["final_decision"] = "dedub"
        return state

    review_text = state["review_text"]
    review_title = state.get("review_title")
    risk_type = current_risk.get("risk_type")
    risk_description = current_risk.get("description")
    risk_factors = current_risk.get("risk_factors")
    risk_factors_str = ", ".join(risk_factors) if isinstance(risk_factors, list) else (str(risk_factors or ""))
    implications = current_risk.get("implications")
    implications_str = ", ".join(implications) if isinstance(implications, list) else (str(implications or ""))

    user_content = CRITIC_USER_TEMPLATE.format(
        review_title=(review_title or "").strip(),
        review_text=(review_text or "").strip(),
        risk_type=(risk_type or "").strip(),
        risk_description=(risk_description or "").strip(),
        risk_factors=(risk_factors_str or "").strip(),
        implications=(implications_str or "").strip(),
    )
    messages = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=user_content),
    ]
    try:
        result: CriticResponse = await structured_critic_llm.ainvoke(messages)
        decision = result.final_decision
        comment = result.comment or ""
    except Exception as e:
        logger.warning("Ошибка критика: %s", e, exc_info=True)
        decision = "complete"
        comment = ""

    if decision == "generate" and critic_iter < max_iter:
        state["final_decision"] = "generator"
        state["critic_comment"] = comment
        state["critic_iter"] = critic_iter + 1
        logger.info("Отзыв %s — Критик: возврат в генератор (заход %s), комментарий: %s", review_id, critic_iter + 1, (comment or "")[:80])
    else:
        state["final_decision"] = "dedub"
        state["critic_comment"] = ""
        logger.info("Отзыв %s — Критик: принято, отправка в дедубликацию", review_id)
    return state


def route_after_critic(state: PipelineState) -> Literal["generator", "dedub"]:
    res = state.get("final_decision")
    if res == "generator":
        return "generator"
    return "dedub"


async def dedub_agent(state: PipelineState) -> PipelineState:
    review_id = state.get("review_id")
    logger.info("Отзыв %s — Дедубликация: поиск существующего или создание нового риска", review_id)

    current_risk = state.get("current_risk")
    known_risks = state.get("known_risks") or []
    known_ids = {int(r["id"]) for r in known_risks if r.get("id") is not None}

    if not current_risk:
        logger.info("Отзыв %s — Дедубликация: риска нет в state, будет создан новый", review_id)
        state["resolved_risk_id"] = None
        state["is_new_risk"] = True
        state["description"] = ""
        state["risk_factors"] = []
        state["implications"] = []
        return state

    def _str(val):
        if val is None:
            return ""
        if isinstance(val, list):
            return " ".join(str(x).strip() for x in val if x).strip()
        return str(val).strip()

    def _to_factors_list(val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, list):
            return [str(x).strip() for x in val if x and str(x).strip()]
        s = str(val).strip()
        return [s] if s else []

    risk_id = current_risk.get("risk_id")
    if risk_id is not None and int(risk_id) in known_ids:
        logger.info("Отзыв %s — Дедубликация: выбран существующий риск из списка известных, risk_id=%s", review_id, risk_id)
        state["resolved_risk_id"] = int(risk_id)
        state["is_new_risk"] = False
        state["description"] = _str(current_risk.get("description"))
        state["risk_factors"] = _to_factors_list(current_risk.get("risk_factors"))
        state["implications"] = _to_factors_list(current_risk.get("implications"))
        return state

    risk_type = _str(current_risk.get("risk_type"))
    description = _str(current_risk.get("description"))
    factors_list = _to_factors_list(current_risk.get("risk_factors"))
    implications_list = _to_factors_list(current_risk.get("implications"))

    if known_risks:
        try:
            catalog_for_llm = [
                {
                    "id": r.get("id"),
                    "risk_type": r.get("risk_type", ''),
                    "description": r.get("description", ''),
                    "risk_factors": r.get("risk_factors", []),
                    "implications": r.get("implications", []),
                }
                for r in known_risks
            ]
            old_risks_catalog = json.dumps(catalog_for_llm, ensure_ascii=False, indent=2)
            new_risk_payload = {
                "risk_type": risk_type,
                "description": description,
                "risk_factors": factors_list,
                "implications": implications_list,
            }
            new_risk_text = json.dumps(new_risk_payload, ensure_ascii=False, indent=2)
            messages = [
                SystemMessage(content=DEDUB_SYSTEM),
                HumanMessage(content=DEDUB_USER_TEMPLATE.format(
                    old_risks_catalog=(old_risks_catalog or "").strip(),
                    new_risk=(new_risk_text or "").strip(),
                )),
            ]
            dedub_result: DedubResponse = await structured_dedub_llm.ainvoke(messages)
            if (
                dedub_result.match_risk_id is not None
                and dedub_result.match_risk_id in known_ids
                and not dedub_result.is_new
            ):
                logger.info(
                    "Отзыв %s — Дедубликация: LLM нашёл дубликат, risk_id=%s",
                    review_id, dedub_result.match_risk_id,
                )
                state["resolved_risk_id"] = dedub_result.match_risk_id
                state["is_new_risk"] = False
                state["description"] = _str(current_risk.get("description"))
                state["risk_factors"] = _to_factors_list(current_risk.get("risk_factors"))
                state["implications"] = _to_factors_list(current_risk.get("implications"))
                return state
        except Exception as e:
            logger.warning("Ошибка LLM-дедубликации: %s", e)

    logger.info("Отзыв %s — Дедубликация: дубликат не найден, будет создан новый риск", review_id)
    state["resolved_risk_id"] = None
    state["is_new_risk"] = True
    state["description"] = _str(current_risk.get("description"))
    state["risk_factors"] = _to_factors_list(current_risk.get("risk_factors"))
    state["implications"] = _to_factors_list(current_risk.get("implications"))
    return state

workflow = StateGraph(PipelineState)

workflow.add_node("generator", generator_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("dedub", dedub_agent)

workflow.add_edge(START, "generator")
workflow.add_edge("generator", "critic")
workflow.add_conditional_edges("critic", route_after_critic, {"generator": "generator", "dedub": "dedub"})
workflow.add_edge("dedub", END)

graph = workflow.compile()

def _review_text_for_risk_search(review: models.Review) -> str:
    """Текст отзыва для эмбеддинга с промптом поиска рисков (тот же формат, что при сохранении)."""
    bank_name = safe_prompt_str(review.bank.name if review.bank else None)
    theme = safe_prompt_str(review.title)
    body = safe_prompt_str(review.text)
    rating_part = ""
    if review.rating is not None and 1 <= review.rating <= 5:
        rating_part = f"Отзыв клиента (оценка 1–5): {int(review.rating)}. "
    return f"Отзыв о банке {bank_name}. {rating_part}Тема {theme}. {body}"


async def select_top_existing_risks_for_review(
    db: AsyncSession,
    review: models.Review,
    top_k: int | None = None,
) -> list[models.Risk]:
    settings = get_settings()
    if top_k is None:
        top_k = settings.llm_top_k_risks

    # Загружаем отзыв с банком для текста; вектор считаем с промптом «поиск рисков»
    rev_result = await db.execute(
        select(models.Review)
        .where(models.Review.id == review.id)
        .options(selectinload(models.Review.bank))
    )
    review_loaded = rev_result.scalar_one_or_none()
    if not review_loaded:
        return []
    review_text = _review_text_for_risk_search(review_loaded)
    embedder = get_embedder()
    review_vectors = await asyncio.to_thread(
        lambda: embedder.embed([review_text], prompt=INSTRUCT_RISK_SEARCH_PROMPT)
    )
    if not review_vectors:
        return []
    review_vec_list = review_vectors[0]

    qdrant_hits = await search_risks_by_vector_async(review_vec_list, top_k)
    if qdrant_hits:
        risk_ids = [rid for rid, _ in qdrant_hits]
        result = await db.execute(select(models.Risk).where(models.Risk.id.in_(risk_ids)))
        all_r = result.scalars().all()
        risks_by_id = {r.id: r for r in all_r}
        return [risks_by_id[rid] for rid in risk_ids if rid in risks_by_id]

    result = await db.execute(select(models.Risk))
    all_risks = result.scalars().all()
    if not all_risks:
        return []

    risk_texts = [
        format_risk_for_embed(
            r.risk_type,
            r.description or "",
            r.risk_factors or [],
        )
        for r in all_risks
    ]
    risk_vecs = np.array(
        await asyncio.to_thread(embedder.embed, risk_texts),
        dtype=float,
    )
    review_vec = np.array(review_vec_list, dtype=float)
    sims = risk_vecs @ review_vec
    if sims.size == 0:
        return []
    top_k_actual = min(top_k, sims.shape[0])
    top_indices = np.argsort(-sims)[:top_k_actual]
    return [all_risks[i] for i in top_indices]

async def score_review_criticality(
    review_title: Optional[str],
    review_text: str,
    risk_type: str,
    risk_description: str,
    risk_factors: str = "",
    implications: str = "",
) -> Optional[int]:
    review_title = (review_title or "Без темы").strip()
    user_content = CRITICALITY_USER_TEMPLATE.format(
        review_title=review_title,
        review_text=(review_text or "").strip(),
        risk_type=(risk_type or "").strip(),
        risk_description=(risk_description or "").strip(),
        risk_factors=(risk_factors or "").strip(),
        implications=(implications or "").strip(),
    )
    messages = [
        SystemMessage(content=CRITICALITY_SYSTEM),
        HumanMessage(content=user_content),
    ]
    try:
        result: ReviewCriticalityResponse = await structured_criticality_llm.ainvoke(messages)
        return result.criticality_score
    except Exception as e:
        logger.warning("Ошибка оценки критичности отзыва: %s", e)
        return None


async def generate_risk_for_review_async(db: AsyncSession, review: models.Review) -> Optional[models.ReviewRisk]:
    if review.sentiment != "negative":
        logger.debug("Отзыв %s не является отрицательным (sentiment=%s), пропускаем", review.id, review.sentiment)
        return None
    vec = await get_vector_for_review_async(review.id)
    if not vec:
        logger.debug("У отзыва %s нет вектора в Qdrant, пропускаем", review.id)
        return None
    # Повторная проверка: отзыв мог уже получить риск в параллельном запуске (до блокировки на API)
    _existing = await db.execute(
        select(models.ReviewRisk).where(models.ReviewRisk.review_id == review.id).limit(1)
    )
    if _existing.scalar_one_or_none():
        logger.info("Отзыв %s уже имеет привязанный риск, пропуск (избегаем дубля)", review.id)
        return None

    bank_name = review.bank.name if review.bank else "Неизвестный банк"
    settings = get_settings()
    known_risks_orm = await select_top_existing_risks_for_review(db, review, top_k=settings.llm_top_k_risks)
    known_risks = [
        {
            "id": r.id,
            "risk_type": r.risk_type,
            "description": r.description,
            "risk_factors": (
                r.risk_factors
                if isinstance(r.risk_factors, list)
                else ([str(r.risk_factors)] if r.risk_factors else [])
            ),
            "implications": (
                r.implications
                if isinstance(r.implications, list)
                else ([str(r.implications)] if r.implications else [])
            ),
        }
        for r in known_risks_orm
    ]

    initial_state: PipelineState = {
        "review_text": review.text,
        "review_title": review.title,
        "bank_name": bank_name,
        "known_risks": known_risks,
        "review_id": review.id,
        "bank_id": review.bank_id,
        "current_risk": None,
        "critic_iter": 0,
        "critic_comment": "",
        "final_decision": "",
        "resolved_risk_id": None,
        "is_new_risk": False,
        "description": "",
        "risk_factors": [],
        "implications": [],
    }

    logger.info("Отзыв %s — Старт пайплайна (Generator → Critic → Dedub)", review.id)
    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("Ошибка пайплайна рисков для отзыва %s: %s", review.id, e, exc_info=True)
        return None

    resolved_risk_id = final_state.get("resolved_risk_id")
    is_new_risk = final_state.get("is_new_risk", True)
    logger.info("Отзыв %s — Пайплайн завершён: risk_id=%s, новый_риск=%s", review.id, resolved_risk_id, is_new_risk)
    description = final_state.get("description") or ""
    risk_factors_list: List[str] = final_state.get("risk_factors") or []
    risk_factors_list = [str(x).strip() for x in risk_factors_list if x and str(x).strip()]
    implications_list: List[str] = final_state.get("implications") or []
    implications_list = [str(x).strip() for x in implications_list if x and str(x).strip()]
    current_risk = final_state.get("current_risk") or {}

    review_date = review.published_at or review.scraped_at
    result = await db.execute(
        select(models.ReviewRisk).where(models.ReviewRisk.review_id == review.id).limit(1)
    )
    existing = result.scalar_one_or_none()

    if is_new_risk:
        risk_type = current_risk.get("risk_type")
        new_risk = models.Risk(
            risk_type=risk_type,
            description=description,
            risk_factors=risk_factors_list or None,
            implications=implications_list or None,
        )
        db.add(new_risk)
        await db.flush()
        resolved_risk_id = new_risk.id
        # Эмбеддинг сохраняем после commit ReviewRisk, чтобы не осталось рисков без привязки к отзыву

    risk_id = int(resolved_risk_id) if resolved_risk_id is not None else None
    if risk_id is None:
        logger.warning("Пайплайн не вернул risk_id для отзыва %s", review.id)
        return None

    if not is_new_risk and risk_id is not None:
        _result = await db.execute(select(models.Risk).where(models.Risk.id == risk_id).limit(1))
        _risk_row = _result.scalar_one_or_none()
        if _risk_row:
            existing_factors = [
                str(x).strip() for x in (_risk_row.risk_factors or []) if x and str(x).strip()
            ]
            new_factors = [f for f in risk_factors_list if f and f not in existing_factors]

            existing_implications = [
                str(x).strip() for x in (_risk_row.implications or []) if x and str(x).strip()
            ]
            new_implications = [i for i in implications_list if i and i not in existing_implications]

            need_update = bool(new_factors or new_implications)
            if new_factors:
                merged_factors = existing_factors + new_factors
                _risk_row.risk_factors = merged_factors
            if new_implications:
                merged_implications = existing_implications + new_implications
                _risk_row.implications = merged_implications

            if need_update:
                db.add(_risk_row)
                await db.flush()
                factors_str = ", ".join(_risk_row.risk_factors or [])
                await asyncio.to_thread(
                    _store_risk_embedding,
                    risk_id,
                    _risk_row.risk_type or "",
                    _risk_row.description or "",
                    factors_str,
                )
                logger.info(
                    "Отзыв %s — Слияние в риск %s: +%s факторов, +%s последствий",
                    review.id, risk_id, len(new_factors), len(new_implications),
                )

    if existing:
        existing.risk_id = risk_id
        existing.review_date = review_date
        await db.commit()
        await db.refresh(existing)
        rr = existing
    else:
        try:
            rr = models.ReviewRisk(
                bank_id=review.bank_id,
                review_id=review.id,
                risk_id=risk_id,
                review_date=review_date,
            )
            db.add(rr)
            await db.commit()
            await db.refresh(rr)
        except IntegrityError:
            await db.rollback()
            result = await db.execute(
                select(models.ReviewRisk).where(models.ReviewRisk.review_id == review.id).limit(1)
            )
            rr = result.scalar_one_or_none()
            if rr:
                rr.risk_id = risk_id
                rr.review_date = review_date
                await db.commit()
                await db.refresh(rr)
            else:
                logger.error("Не удалось создать/обновить ReviewRisk для отзыва %s", review.id)
                return None

    # Сохраняем эмбеддинг нового риска только после успешной привязки к отзыву (commit уже выполнен)
    if is_new_risk:
        factors_str = ", ".join(risk_factors_list) if risk_factors_list else ""
        await asyncio.to_thread(
            _store_risk_embedding,
            risk_id,
            (current_risk.get("risk_type") or ""),
            description or "",
            factors_str,
        )

    result = await db.execute(select(models.Risk).where(models.Risk.id == risk_id).limit(1))
    risk_row = result.scalar_one_or_none()
    risk_type_str = risk_row.risk_type if risk_row else ""
    risk_desc = (risk_row.description or "") if risk_row else description
    if risk_row and risk_row.risk_factors is not None:
        risk_factors_for_scorer = (
            ", ".join(risk_row.risk_factors)
            if isinstance(risk_row.risk_factors, list)
            else str(risk_row.risk_factors)
        )
    else:
        risk_factors_for_scorer = ", ".join(risk_factors_list) if risk_factors_list else ""

    if risk_row and risk_row.implications is not None:
        implications_for_scorer = (
            ", ".join(risk_row.implications)
            if isinstance(risk_row.implications, list)
            else str(risk_row.implications)
        )
    else:
        implications_for_scorer = ", ".join(implications_list) if implications_list else ""

    logger.info("Отзыв %s — Оценка критичности отзыва (1–5)", review.id)
    score = await score_review_criticality(
        review_title=review.title,
        review_text=review.text,
        risk_type=risk_type_str,
        risk_description=risk_desc,
        risk_factors=risk_factors_for_scorer,
        implications=implications_for_scorer,
    )
    if score is not None:
        review.criticality_score = score
        await db.commit()
        logger.info("Отзыв %s — Критичность отзыва: %s", review.id, score)
    else:
        logger.warning("Отзыв %s — Оценка критичности не получена", review.id)

    return rr


def _store_risk_embedding(
    risk_id: int,
    risk_type: str,
    description: str = "",
    risk_factors: str = "",
) -> None:
    text = format_risk_for_embed(risk_type, description, risk_factors)
    try:
        embedder = get_embedder()
        vectors = embedder.embed([text])
        if vectors:
            upsert_risk_vector(risk_id, vectors[0], title=(risk_type or "")[:500])
    except Exception as e:
        logger.warning("Не удалось сохранить эмбеддинг риска %s: %s", risk_id, e)
