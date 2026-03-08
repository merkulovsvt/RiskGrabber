import datetime as dt
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from .config import get_settings
from . import models

logger = logging.getLogger(__name__)

# Кэш списка прокси (перечитываем при каждом создании сессии или можно раз в N минут)
_proxy_list: Optional[List[str]] = None
_proxy_list_mtime: float = 0


class ScraperForbiddenError(Exception):
    """Поднятие при 403 Forbidden — скрапер должен сразу остановиться."""
    def __init__(self, message: str, page: Optional[int] = None) -> None:
        self.page = page
        super().__init__(message)

# User-Agent ротация для обхода блокировок
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]


def _load_proxies() -> List[str]:
    """Читает прокси из папки: все .txt файлы, одна строка = один прокси (пустые и # пропускаем)."""
    global _proxy_list, _proxy_list_mtime
    settings = get_settings()
    if settings.proxies_dir:
        proxy_dir = Path(settings.proxies_dir)
    else:
        # По умолчанию: RiskGrabber/proxies (рядом с backend)
        proxy_dir = Path(__file__).resolve().parent.parent / "proxies"
    if not proxy_dir.is_dir():
        _proxy_list = []
        _proxy_list_mtime = 0
        return []
    try:
        mtime = max((f.stat().st_mtime for f in proxy_dir.iterdir() if f.suffix.lower() == ".txt"), default=0)
        if _proxy_list is not None and mtime <= _proxy_list_mtime:
            return _proxy_list
    except OSError:
        _proxy_list = []
        return []
    out: List[str] = []
    for f in sorted(proxy_dir.glob("*.txt")):
        try:
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    out.append(s)
        except OSError as e:
            logger.warning("Не удалось прочитать файл прокси %s: %s", f, e)
    _proxy_list = out
    _proxy_list_mtime = mtime
    if out:
        logger.info("Загружено %s прокси из %s", len(out), proxy_dir)
    return out


def _get_session() -> requests.Session:
    """
    Создаёт сессию с новым случайным User-Agent и случайным прокси (если есть).
    Вызывать на каждый запрос, чтобы каждый запрос шёл с новым UA и прокси.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    })
    proxies = _load_proxies()
    if proxies:
        url = random.choice(proxies).strip()
        if url and not url.startswith("http"):
            url = "http://" + url
        if url:
            session.proxies = {"http": url, "https": url}
    return session


def _parse_review_detail_page(review_url: str) -> Optional[Tuple[str, Optional[float], str, Optional[dt.datetime]]]:
    """
    Парсит страницу конкретного отзыва. На каждый вызов — новый запрос с новым
    случайным User-Agent и прокси.
    
    Args:
        review_url: URL страницы отзыва (например, /services/responses/bank/response/12972986/)
    
    Returns:
        (bank_name, rating, text, published_at) или None при ошибке
    """
    base_url = "https://www.banki.ru"
    full_url = base_url + review_url if review_url.startswith("/") else review_url

    try:
        # Задержка перед запросом страницы отзыва (2–6 сек)
        time.sleep(random.uniform(2.0, 6.0))

        session = _get_session()
        resp = session.get(full_url, timeout=30)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        logger.debug("=== Парсинг отзыва: %s ===", review_url)
        
        # 1. Название банка из alt атрибута изображения
        bank_name = None
        bank_img = soup.select_one("img.lazy-load[alt]")
        if bank_img:
            bank_name = bank_img.get("alt", "").strip()
            logger.debug("Найден банк (через img.lazy-load): %s", bank_name)
        
        # Альтернативный способ: ищем в ссылке на банк
        if not bank_name:
            bank_link = soup.select_one("a[href*='/banks/bank/'] img[alt]")
            if bank_link:
                bank_name = bank_link.get("alt", "").strip()
                logger.debug("Найден банк (через ссылку): %s", bank_name)
        
        if not bank_name:
            logger.warning("Банк не найден на странице %s", review_url)
            return None
        
        # 2. Рейтинг из элемента с классом rating-grade
        rating = None
        rating_el = soup.select_one(".rating-grade")
        if rating_el:
            rating_text = rating_el.get_text(strip=True)
            logger.debug("Найден элемент рейтинга (.rating-grade): текст='%s'", rating_text)
            # Ищем число от 1 до 5
            match = re.search(r'(\d+)', rating_text)
            if match:
                try:
                    rating_val = int(match.group(1))
                    if 1 <= rating_val <= 5:
                        rating = float(rating_val)
                        logger.debug("Извлечён рейтинг: %s", rating)
                except ValueError:
                    pass
        
        # 3. Текст отзыва - используем div.lb1789875 (самый надёжный способ)
        text = None
        content_div = soup.find('div', class_='lb1789875')
        if content_div:
            # Получаем текст, сохраняя абзацы
            review_text = content_div.get_text(separator='\n\n', strip=True)
            # Убираем лишние переносы
            text = re.sub(r'\n{3,}', '\n\n', review_text).strip()
            logger.debug("Извлечённый текст из div.lb1789875: длина=%s, первые 100 символов='%s'", 
                       len(text) if text else 0,
                       text[:100] if text else '')
        
        # Если не нашли через lb1789875, пробуем старые способы
        if not text:
            # Приоритет 1: ищем div с font-size='fs18' и классом bKVLHc
            text_el = soup.select_one("div[font-size='fs18'].bKVLHc")
            if text_el:
                logger.debug("Найден элемент текста по div[font-size='fs18'].bKVLHc")
            else:
                # Приоритет 2: ищем div с font-size='fs18' и классом, содержащим bKVLHc
                text_el = soup.select_one("div[font-size='fs18'][class*='bKVLHc']")
                if text_el:
                    logger.debug("Найден элемент текста по div[font-size='fs18'][class*='bKVLHc']")
                else:
                    # Приоритет 3: ищем div с классом bKVLHc
                    text_el = soup.select_one("div.bKVLHc")
                    if text_el:
                        logger.debug("Найден элемент текста по div.bKVLHc")
                    else:
                        # Приоритет 4: ищем div с font-size='fs18'
                        text_el = soup.select_one("div[font-size='fs18']")
                        if text_el:
                            logger.debug("Найден элемент текста по div[font-size='fs18']")
            
            if text_el:
                # Собираем текст из параграфов <p>
                paragraphs = text_el.find_all("p")
                if paragraphs:
                    logger.debug("Найдено %s параграфов в элементе текста", len(paragraphs))
                    text_parts = []
                    for p in paragraphs:
                        p_text = p.get_text(strip=True)
                        if p_text and len(p_text) > 0:
                            text_parts.append(p_text)
                    text = "\n".join(text_parts)
                else:
                    # Если нет параграфов, берём весь текст
                    logger.debug("Параграфы не найдены, извлекаем весь текст элемента")
                    text = text_el.get_text(separator="\n", strip=True)
                
                # Очищаем от лишних переносов
                if text:
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    text = "\n".join(lines)
        
        if not text:
            logger.warning("Не удалось извлечь текст отзыва из %s", review_url)
            return None
        
        # Проверяем, что текст не является служебным
        service_texts = [
            "Все вклады и счета",
            "Вклады онлайн на Банки.ру",
            "Специальные предложения",
            "Калькулятор вкладов",
            "Накопительные счета",
            "Вклады под высокий процент",
            "Пополняемые вклады",
            "Ставка ЦБ РФ",
            "Микрозаймы",
            "Автокредиты",
            "Кредитный калькулятор",
            "Подбор займа"
        ]
        if any(skip in text for skip in service_texts):
            logger.warning("Извлечённый текст содержит служебную информацию, отбрасываем")
            return None
        
        if len(text) < 20:
            logger.warning("Текст слишком короткий (%s символов), отбрасываем", len(text))
            return None
        
        # 4. Дата публикации
        published_at = None
        date_el = soup.select_one("span.l10fac986")
        if date_el:
            date_text = date_el.get_text(strip=True)
            logger.debug("Найдена дата (span.l10fac986): '%s'", date_text)
            
            # Парсим дату в формате "05.03.2026 16:11"
            date_match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})', date_text)
            if date_match:
                day, month, year, hour, minute = date_match.groups()
                try:
                    published_at = dt.datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute)
                    )
                    logger.debug("Распарсена дата: %s", published_at)
                except ValueError:
                    logger.debug("Не удалось распарсить дату: %s", date_text)
        
        return (bank_name, rating, text, published_at)
        
    except requests.exceptions.RequestException as e:
        logger.error("Ошибка HTTP при парсинге %s: %s", review_url, e)
        return None
    except Exception as e:
        logger.exception("Неожиданная ошибка при парсинге %s: %s", review_url, e)
        return None


def fetch_reviews_page(
    page: int,
    db: Session,
) -> Iterable[Tuple[str, str, Optional[float], Optional[str], str, Optional[dt.datetime]]]:
    """
    Получает список отзывов со страницы списка отзывов.
    Заходит только на те ссылки, external_id которых ещё нет в базе.
    
    Yields:
        (bank_name, external_id, rating, title, text, published_at)
    """
    settings = get_settings()
    base_url = settings.banki_base_url
    url = f"{base_url}&page={page}"

    try:
        logger.info("Загрузка страницы списка отзывов: %s", url)

        # Задержка перед запросом страницы списка (2–6 сек)
        time.sleep(random.uniform(2.0, 6.0))

        session = _get_session()
        resp = session.get(url, timeout=30)
        if resp.status_code == 403:
            raise ScraperForbiddenError("403 Forbidden", page)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Находим все ссылки на отзывы
        review_links = soup.select("a[href*='/services/responses/bank/response/']")
        logger.info("Найдено %s ссылок на отзывы на странице %s", len(review_links), page)
        
        # Собираем уникальные (external_id, review_path, title)
        seen_ids: set[str] = set()
        links_to_parse: List[Tuple[str, str, Optional[str]]] = []
        for link in review_links:
            href = link.get("href")
            if not href:
                continue
            match = re.search(r'/response/(\d+)/?', href)
            if not match:
                continue
            external_id = match.group(1)
            if external_id in seen_ids:
                continue
            seen_ids.add(external_id)
            review_path = f"/services/responses/bank/response/{external_id}/"
            title = link.get_text(strip=True)[:500] if link.get_text(strip=True) else None
            links_to_parse.append((external_id, review_path, title))
        
        if not links_to_parse:
            return
        
        # Какие external_id уже есть в базе — на них не заходим
        ids_on_page = [ext_id for ext_id, _, _ in links_to_parse]
        existing_ids: set[str] = set(
            row[0] for row in db.query(models.Review.external_id).filter(
                models.Review.external_id.in_(ids_on_page),
                models.Review.external_id.isnot(None),
            ).distinct().all()
            if row[0]
        )
        to_fetch = [(ext_id, path, title) for ext_id, path, title in links_to_parse if ext_id not in existing_ids]
        skipped = len(links_to_parse) - len(to_fetch)
        if skipped:
            logger.info("Страница %s: пропущено %s отзывов (уже в базе), заходим в %s", page, skipped, len(to_fetch))
        if not to_fetch:
            logger.info("Страница %s: все отзывы уже в базе", page)
            return

        # Параллельная загрузка страниц отзывов (2 потока), чтобы не блокировать надолго
        PARALLEL_DETAILS = 2
        def fetch_one(ext_id: str, path: str, tit: Optional[str]) -> Optional[Tuple[str, str, Optional[float], Optional[str], str, Optional[dt.datetime]]]:
            res = _parse_review_detail_page(path)
            if res:
                bank_name, rating, text, published_at = res
                return (bank_name, ext_id, rating, tit, text, published_at)
            return None

        with ThreadPoolExecutor(max_workers=PARALLEL_DETAILS) as executor:
            future_to_item = {
                executor.submit(fetch_one, ext_id, path, title): (ext_id, path)
                for ext_id, path, title in to_fetch
            }
            for future in as_completed(future_to_item):
                ext_id, path = future_to_item[future]
                try:
                    row = future.result()
                    if row:
                        yield row
                    else:
                        logger.warning("Не удалось распарсить отзыв %s", path)
                except Exception as e:
                    logger.warning("Ошибка при парсинге отзыва %s: %s", ext_id, e)

    except ScraperForbiddenError:
        raise
    except requests.exceptions.RequestException as e:
        if getattr(e, "response", None) and e.response.status_code == 403:
            raise ScraperForbiddenError("403 Forbidden", page) from e
        logger.error("Ошибка HTTP при загрузке страницы %s: %s", page, e)
    except Exception as e:
        logger.exception("Неожиданная ошибка при загрузке страницы %s: %s", page, e)


def upsert_bank(db: Session, name: str) -> models.Bank:
    """Создаёт или получает банк по имени."""
    bank = db.query(models.Bank).filter(models.Bank.name == name).one_or_none()
    if not bank:
        bank = models.Bank(name=name, slug=name.lower().replace(" ", "-"))
        db.add(bank)
        db.commit()
        db.refresh(bank)
    return bank


def ingest_reviews(
    db: Session,
    max_pages: int = 3,
    progress_callback: Optional[Callable[[str, str, dict], None]] = None,
) -> int:
    """
    Собирает новые отзывы с первых max_pages страниц.

    progress_callback(stage, message, detail) вызывается при прогрессе.
    
    Returns:
        Количество новых отзывов, добавленных в БД.
    """
    new_count = 0

    def report(stage: str, message: str, detail: Optional[dict] = None) -> None:
        if progress_callback:
            progress_callback(stage, message, detail or {})
    
    for page in range(1, max_pages + 1):
        try:
            if progress_callback:
                report("scraping", f"Страница {page} из {max_pages}...", {"page": page, "max_pages": max_pages})
            page_reviews = 0
            for bank_name, external_id, rating, title, text, published_at in fetch_reviews_page(page, db):
                # Проверяем на дубликаты
                bank = db.query(models.Bank).filter(models.Bank.name == bank_name).one_or_none()
                if bank:
                    existing = (
                        db.query(models.Review)
                        .filter(
                            models.Review.external_id == external_id,
                            models.Review.bank_id == bank.id,
                        )
                        .one_or_none()
                    )
                    if existing:
                        logger.debug("Пропущен дубликат: external_id=%s, bank_id=%s", external_id, bank.id)
                        continue
                
                # Создаём или получаем банк
                bank = upsert_bank(db, bank_name)
                
                # Повторная проверка на дубликат
                if external_id:
                    existing = (
                        db.query(models.Review)
                        .filter(
                            models.Review.external_id == external_id,
                            models.Review.bank_id == bank.id,
                        )
                        .one_or_none()
                    )
                    if existing:
                        logger.debug("Пропущен дубликат после создания банка: external_id=%s, bank_id=%s", external_id, bank.id)
                        continue
                
                review = models.Review(
                    bank_id=bank.id,
                    external_id=external_id,
                    rating=rating,
                    title=title or None,
                    text=text,
                    published_at=published_at,
                )
                db.add(review)
                new_count += 1
                page_reviews += 1

            db.commit()
            if progress_callback:
                report("scraping", f"Страница {page}: +{page_reviews} отзывов, всего {new_count}", {"page": page, "page_reviews": page_reviews, "total_new": new_count})
            logger.info("Страница %s: обработано %s новых отзывов", page, page_reviews)
            
        except ScraperForbiddenError as e:
            logger.warning("Остановка парсинга: 403 Forbidden (страница %s). Сайт заблокировал доступ.", e.page)
            if progress_callback:
                report("scraping", f"Остановка: 403 Forbidden на странице {e.page}", {"done": True, "total_new": new_count, "stopped_403": True})
            break
        except Exception as exc:
            logger.exception("Ошибка при обработке страницы %s: %s", page, exc)
            db.rollback()
            time.sleep(random.uniform(2, 6))
    
    if progress_callback:
        report("scraping", f"Парсинг завершён. Собрано отзывов: {new_count}", {"done": True, "total_new": new_count})
    return new_count


def ingest_reviews_since(
    db: Session,
    since: dt.datetime,
    progress_callback: Optional[Callable[[str, str, dict], None]] = None,
) -> int:
    """
    Исторический сбор: собирает отзывы, опубликованные после указанной даты.
    Идёт по страницам от свежих к старым и останавливается, когда встречает отзывы старше since.

    progress_callback(stage, message, detail) вызывается при прогрессе.
    """
    new_count = 0
    page = 1
    max_consecutive_errors = 5
    consecutive_errors = 0
    stop = False

    def report(stage: str, message: str, detail: Optional[dict] = None) -> None:
        if progress_callback:
            progress_callback(stage, message, detail or {})
    
    while not stop:
        if consecutive_errors >= max_consecutive_errors:
            logger.error("Слишком много ошибок подряд. Прерываем исторический сбор.")
            break
        
        try:
            if progress_callback:
                report("scraping", f"Парсинг страницы {page}...", {"page": page, "total_new": new_count})
            # Задержка между страницами (2–6 сек)
            if page > 1:
                delay = random.uniform(2.0, 6.0)
                time.sleep(delay)
            
            page_reviews = 0
            for bank_name, external_id, rating, title, text, published_at in fetch_reviews_page(page, db):
                # Если дата известна и она раньше пороговой — останавливаемся
                if published_at and published_at < since:
                    stop = True
                    logger.info("Достигнута дата %s. Останавливаем сбор.", since)
                    break
                
                # Проверяем на дубликаты
                bank = db.query(models.Bank).filter(models.Bank.name == bank_name).one_or_none()
                if bank:
                    existing = (
                        db.query(models.Review)
                        .filter(
                            models.Review.external_id == external_id,
                            models.Review.bank_id == bank.id,
                        )
                        .one_or_none()
                    )
                    if existing:
                        logger.debug("Пропущен дубликат: external_id=%s, bank_id=%s", external_id, bank.id)
                        continue
                
                # Создаём или получаем банк
                bank = upsert_bank(db, bank_name)
                
                # Повторная проверка на дубликат
                if external_id:
                    existing = (
                        db.query(models.Review)
                        .filter(
                            models.Review.external_id == external_id,
                            models.Review.bank_id == bank.id,
                        )
                        .one_or_none()
                    )
                    if existing:
                        logger.debug("Пропущен дубликат после создания банка: external_id=%s, bank_id=%s", external_id, bank.id)
                        continue
                
                review = models.Review(
                    bank_id=bank.id,
                    external_id=external_id,
                    rating=rating,
                    title=title or None,
                    text=text,
                    published_at=published_at,
                )
                db.add(review)
                new_count += 1
                page_reviews += 1
            
            db.commit()
            consecutive_errors = 0
            if progress_callback:
                report("scraping", f"Страница {page}: +{page_reviews} отзывов, всего {new_count}", {"page": page, "page_reviews": page_reviews, "total_new": new_count})
            logger.info("Страница %s: обработано %s новых отзывов", page, page_reviews)
            
            if stop:
                break
            
            page += 1
            
        except ScraperForbiddenError as e:
            logger.warning("Остановка парсинга: 403 Forbidden (страница %s). Сайт заблокировал доступ.", e.page)
            if progress_callback:
                report("scraping", f"Остановка: 403 Forbidden на странице {e.page}", {"done": True, "total_new": new_count, "stopped_403": True})
            break
            
        except requests.exceptions.HTTPError as e:
            consecutive_errors += 1
            if e.response.status_code == 403:
                logger.warning("Остановка парсинга: 403 Forbidden на странице %s.", page)
                if progress_callback:
                    report("scraping", f"Остановка: 403 Forbidden на странице {page}", {"done": True, "total_new": new_count, "stopped_403": True})
                break
            else:
                logger.exception("HTTP error while backfill-scraping page %s: %s", page, e)
            db.rollback()
            time.sleep(random.uniform(3, 6))
            
        except Exception as exc:
            consecutive_errors += 1
            logger.exception("Ошибка при обработке страницы %s: %s", page, exc)
            db.rollback()
            if consecutive_errors >= max_consecutive_errors:
                logger.error("Слишком много ошибок подряд. Прерываем исторический сбор.")
                break
            time.sleep(random.uniform(2, 6))
            page += 1

    if progress_callback:
        report("scraping", f"Парсинг завершён. Собрано отзывов: {new_count}", {"done": True, "total_new": new_count})
    return new_count
