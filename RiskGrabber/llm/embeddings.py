import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..backend.config import get_settings


def _embed_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

_RISKGRABBER_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _RISKGRABBER_ROOT.parent
_DEFAULT_LOCAL_MODEL_DIR = _PROJECT_ROOT / "data" / "models" / "qwen3-embedding-0.6b"
_HF_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


INSTRUCT_BANK_REVIEW_TASK = "Дан отзыв о банке, представь его для бинарной классификации тональности (позитив или негатив)"
INSTRUCT_BANK_REVIEW_PROMPT = f"Instruct: {INSTRUCT_BANK_REVIEW_TASK}\nQuery: "

INSTRUCT_RISK_SEARCH_TASK = "Дан отзыв о банке, представь его для поиска подходящих рисков по смыслу (вид риска, описание события, факторы)"
INSTRUCT_RISK_SEARCH_PROMPT = f"Instruct: {INSTRUCT_RISK_SEARCH_TASK}\nQuery: "


def format_risk_for_embed(risk_type: str, description: str, risk_factors: str | list) -> str:
    risk_type = (risk_type or "").strip()
    description = (description or "").strip()
    if isinstance(risk_factors, list):
        factors_str = ", ".join((x or "").strip() for x in risk_factors if (x or "").strip())
    else:
        factors_str = (risk_factors or "").strip()
    parts = []
    if risk_type:
        parts.append(f"Вид риска: {risk_type}")
    if description:
        parts.append(f"Описание события: {description}")
    if factors_str:
        parts.append(f"Факторы риска: {factors_str}")
    return ". ".join(parts) if parts else "Риск"


logger = logging.getLogger(__name__)
_embedder: Optional["LocalHFEmbedder"] = None


class LocalHFEmbedder:
    def __init__(self, model_name: str) -> None:
        device = _embed_device()
        logger.info("Embedder device: %s", device)
        self.model = SentenceTransformer(
            model_name,
            tokenizer_kwargs={"padding_side": "left", 'attn_implementation': 'sdpa'},
            device=device,
            trust_remote_code=True,
        )

    def embed(
        self,
        texts: Iterable[str],
        *,
        prompt: Optional[str] = None,
        prompt_name: Optional[str] = None,
    ) -> List[List[float]]:
        texts_list = list(texts)
        kwargs = dict(
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if prompt is not None:
            kwargs["prompt"] = prompt
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        try:
            vectors = self.model.encode(texts_list, **kwargs)
            return [vec.astype(float).tolist() for vec in np.atleast_2d(vectors)]
        except Exception as e:
            if len(texts_list) <= 1:
                raise
            logger.warning(
                "Batch embed failed (n=%d), falling back to per-item encode: %s",
                len(texts_list),
                e,
            )
            result: List[List[float]] = []
            for t in texts_list:
                vec = self.model.encode([t], **kwargs)
                row = np.atleast_2d(vec)[0]
                result.append(row.astype(float).tolist())
            return result


def _ensure_local_model() -> str:
    if _DEFAULT_LOCAL_MODEL_DIR.is_dir() and any(_DEFAULT_LOCAL_MODEL_DIR.iterdir()):
        return str(_DEFAULT_LOCAL_MODEL_DIR)
    _DEFAULT_LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from HuggingFace and saving to %s", _HF_MODEL_ID, _DEFAULT_LOCAL_MODEL_DIR)
    model = SentenceTransformer(
        _HF_MODEL_ID,
        tokenizer_kwargs={"padding_side": "left", 'attn_implementation': 'sdpa'},
        device=_embed_device(),
        trust_remote_code=True,
    )
    model.save(str(_DEFAULT_LOCAL_MODEL_DIR))
    logger.info("Model saved to %s", _DEFAULT_LOCAL_MODEL_DIR)
    return str(_DEFAULT_LOCAL_MODEL_DIR)


def _resolve_embed_model_name() -> str:
    settings = get_settings()
    if settings.embed_model_path:
        path = Path(settings.embed_model_path)
        if not path.is_absolute():
            path = (_PROJECT_ROOT / path).resolve()
        return str(path)
    if settings.hf_model_name == _HF_MODEL_ID:
        return _ensure_local_model()
    return settings.hf_model_name


def get_embedder() -> LocalHFEmbedder:
    global _embedder
    if _embedder is None:
        model_name = _resolve_embed_model_name()
        logger.info("Loading embedder model (once): %s", model_name)
        _embedder = LocalHFEmbedder(model_name)
        logger.info("Embedder model loaded and cached in memory")
    return _embedder

