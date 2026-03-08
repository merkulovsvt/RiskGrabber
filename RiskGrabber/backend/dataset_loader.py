import datetime as dt
import logging
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

try:
    from datasets import Dataset, load_dataset
except ImportError:
    Dataset = None
    load_dataset = None

from .config import get_settings
from . import models
from .time_utils import moscow_now

logger = logging.getLogger(__name__)

# Seed для воспроизводимого распределения дат
_DISTRIBUTION_SEED = 42


def _date_range(start: dt.datetime, end: dt.datetime):
    """Все даты от start до end включительно."""
    curr = start
    while curr <= end:
        yield curr
        curr += dt.timedelta(days=1)


def _smooth_weights(weights: np.ndarray, sigma: int = 3) -> np.ndarray:
    """Простое сглаживание скользящим средним (без scipy)."""
    n = len(weights)
    if n == 0 or sigma <= 0:
        return weights
    out = np.zeros_like(weights)
    for i in range(n):
        lo = max(0, i - sigma)
        hi = min(n, i + sigma + 1)
        out[i] = float(np.mean(weights[lo:hi]))
    return out


def _redistribute_dates_deterministic(
    rows: List[dict],
    parse_date,
) -> None:
    """
    Перераспределяет даты в rows детерминированно по неравномерному распределению
    между min и max датой из данных. Порядок строк не меняется (без shuffle).
    Модифицирует rows in-place, подменяя review_dttm.
    """
    dates_parsed = [parse_date(r.get("review_dttm")) for r in rows]
    valid = [(i, d) for i, d in enumerate(dates_parsed) if d is not None]
    if not valid:
        return
    min_date = min(d for _, d in valid)
    max_date = max(d for _, d in valid)
    if min_date > max_date:
        return
    all_dates = list(_date_range(min_date, max_date))
    n_days = len(all_dates)
    if n_days == 0:
        return
    total_records = len(rows)

    random.seed(_DISTRIBUTION_SEED)
    np.random.seed(_DISTRIBUTION_SEED)

    base_weights = np.random.uniform(0.5, 1.5, n_days)
    smoothed_weights = _smooth_weights(base_weights, sigma=3)
    days_array = np.arange(n_days, dtype=float)
    seasonal_pattern = 1.0 + 0.3 * np.sin(2 * np.pi * days_array / 30)
    final_weights = smoothed_weights * seasonal_pattern
    final_weights = np.abs(final_weights)
    probabilities = final_weights / final_weights.sum()

    counts = np.random.multinomial(total_records, probabilities)
    date_slots: List[dt.datetime] = []
    for i, count in enumerate(counts):
        d = all_dates[i]
        date_slots.extend([d] * int(count))
    if len(date_slots) != total_records:
        delta = total_records - len(date_slots)
        if delta > 0:
            date_slots.extend([all_dates[-1]] * delta)
        else:
            date_slots = date_slots[:total_records]

    for i, row in enumerate(rows):
        if i < len(date_slots):
            row["review_dttm"] = date_slots[i].strftime("%Y-%m-%d")


def _get_cache_dir() -> Path:
    settings = get_settings()
    if settings.dataset_cache_dir:
        return Path(settings.dataset_cache_dir)
    root = Path(__file__).resolve().parent.parent.parent
    return root / "data" / "datasets"


def load_russian_bank_reviews_into_db(db: Session, limit: int = 50) -> dict:
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if load_dataset is None:
        raise RuntimeError("Установите пакет datasets: pip install datasets")
    ds = load_dataset(
        "Romjiik/Russian_bank_reviews",
        cache_dir=str(cache_dir),
    )
    # Обычно это DatasetDict с ключом "train"
    if "train" in ds:
        data = ds["train"]
    else:
        data = ds[list(ds.keys())[0]]

    inserted = 0
    skipped = 0
    errors = 0

    def upsert_bank(name: str) -> Optional[models.Bank]:
        name = (name or "").strip()
        if not name:
            return None
        bank = db.query(models.Bank).filter(models.Bank.name == name).one_or_none()
        if not bank:
            bank = models.Bank(name=name, slug=name.lower().replace(" ", "-")[:255])
            db.add(bank)
            db.commit()
            db.refresh(bank)
        return bank

    def parse_date(val) -> Optional[dt.datetime]:
        """Парсит дату из датасета. Вход приходит как yyyy-mm-dd (напр. 2022-01-12). Записываем в БД как есть."""
        if val is None:
            return None
        if isinstance(val, dt.datetime):
            return val
        if isinstance(val, str):
            val = val.strip()[:10]
            for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
                try:
                    return dt.datetime.strptime(val, fmt).replace(tzinfo=None)
                except ValueError:
                    continue
        return None

    # Сортировка по дате (именно по дате, а не по строке/числу): старые → новые
    rows = [dict(data[i]) for i in range(len(data))]
    rows.sort(key=lambda r: (parse_date(r.get("review_dttm")) or dt.datetime.min))

    # Детерминированное неравномерное распределение дат по диапазону (без shuffle, порядок строк сохраняется)
    _redistribute_dates_deterministic(rows, parse_date)

    for idx, row in enumerate(rows):
        if inserted >= limit:
            break
        try:
            bank_name = row.get("bank")
            if not bank_name:
                errors += 1
                continue
            bank = upsert_bank(str(bank_name).strip())
            if not bank:
                errors += 1
                continue

            external_id = "hf_" + str(row.get("Unnamed: 0", idx))
            existing = (
                db.query(models.Review)
                .filter(
                    models.Review.external_id == external_id,
                    models.Review.bank_id == bank.id,
                )
                .one_or_none()
            )
            if existing:
                skipped += 1
                continue

            review_text = row.get("review")
            if review_text is None or (isinstance(review_text, str) and not review_text.strip()):
                skipped += 1
                continue

            title = row.get("review_title")
            if title is not None:
                title = str(title).strip()[:500] or None
            else:
                title = None

            rating_val = row.get("rating_value")
            if rating_val is not None:
                try:
                    r = int(rating_val)
                    rating = float(r) if 1 <= r <= 5 else None
                except (TypeError, ValueError):
                    rating = None
            else:
                rating = None

            published_at = parse_date(row.get("review_dttm"))

            review = models.Review(
                bank_id=bank.id,
                external_id=external_id,
                rating=rating,
                title=title,
                text=str(review_text).strip()[:500000],
                published_at=published_at,
                scraped_at=moscow_now(),
            )
            db.add(review)
            db.commit()
            inserted += 1
        except Exception as e:
            logger.warning("Ошибка при импорте строки %s: %s", idx, e)
            db.rollback()
            errors += 1

    return {
        "inserted": inserted,
        "skipped_duplicates": skipped,
        "errors": errors,
        "cache_dir": str(cache_dir),
    }
