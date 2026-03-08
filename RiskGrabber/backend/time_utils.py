"""
Московское время для хранения дат в БД.
Все даты в БД интерпретируются как московское время (naive datetime).
Используется UTC+3 без zoneinfo, чтобы работало на Windows без пакета tzdata.
"""
import datetime as dt

# Москва: UTC+3 (без перехода на летнее время с 2011)
_MOSCOW_OFFSET = dt.timedelta(hours=3)


def moscow_now() -> dt.datetime:
    """Текущее время в Москве, без timezone (naive) — для сохранения в БД."""
    utc_now = dt.datetime.now(dt.timezone.utc)
    moscow = (utc_now + _MOSCOW_OFFSET).replace(tzinfo=None)
    return moscow
